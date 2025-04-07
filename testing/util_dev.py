import re, unicodedata, pickle, tiktoken, boto3, os, botocore, base64, io
from copy import deepcopy
from pymilvus import MilvusClient
from scipy.sparse import dok_matrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from pyroaring import BitMap
from scipy.sparse import dok_matrix
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv() #for local development
from openai import AzureOpenAI
from openai._types import NOT_GIVEN
from openai._exceptions import RateLimitError
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
import pathlib, boto3, botocore, json, os, pickle, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from sentence_transformers import SentenceTransformer


# WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute() 
WORKING_DIRECTORY = os.getcwd()



# ---------- SEMANTIC SETUP ----------

print("SEMANTIC SETUP")

vectorizer = SentenceTransformer('BAAI/bge-base-en-v1.5')
vectors = pickle.load(open(WORKING_DIRECTORY + "semantic_vectors.bin", "rb"))


# ---------- LLM SETUP ----------

print("LLM SETUP")

# load prompts
validate_query_prompt = open(WORKING_DIRECTORY + "/prompts/validate_query.prompt", "r", encoding="utf-8").read()
make_answer_prompt = open(WORKING_DIRECTORY + "/prompts/make_answer.prompt", "r", encoding="utf-8").read()
combine_answers_prompt = open(WORKING_DIRECTORY + "/prompts/combine_answers.prompt", "r", encoding="utf-8").read()


#load GPT tokenizer
gpt_tokenizer = tiktoken.get_encoding("cl100k_base")


gpt35_client = AzureOpenAI(
    api_key=os.environ["OPENAI_35_KEY"],
    api_version="2024-02-15-preview",   
    azure_endpoint="https://agco-can-sandbox.openai.azure.com/")

gpt4o_client = AzureOpenAI(
        api_key=os.environ["OPENAI_4o_KEY"],
        api_version="2024-02-15-preview",
        azure_endpoint="https://agco-sandbox.openai.azure.com/")


"""
Send a message to the GPT model and return the response
Input: list of dictionaries representing the messages
Output: string containing the response
"""
@retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
def get_gpt_response(messages, model="gpt4o", force_json=False, temp=0.5, seed=42, max_tok=4096):

    if model == "gpt4o": client = gpt4o_client
    elif model == "gpt35": client = gpt35_client
    else: raise ValueError(f"Invalid model: {model}")

    try:

        resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"} if force_json else NOT_GIVEN,
                temperature=temp,
                seed=seed,
                timeout=300,
                max_tokens=max_tok)

        return resp.choices[0].message.content
        

    except RateLimitError as e:
        print(f"GPT RATE LIMIT ERROR: {e}")
        raise

    except Exception as e:
        print(f"GPT ERROR: {e}")
        raise




# ---------- MAIN FUNCTIONS ----------

def search_sources(query, filters):
    
    query_vector = vectorizer.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_vector, vectors)
    similarities = np.squeeze(similarities)
    N = 10
    top_N = similarities.argsort()[-N:][::-1]

    sources = []
    for i in top_N:
        source = {}
        source['show_name'] = filters[i]['show_name']
        source['episode_name'] = filters[i]['episode_name']
        source['content'] = filters[i]['content']
        source['similarity'] = similarities[i]
        sources.append(source)

    return sources


def make_answer(source, query):

    sys_txt = make_answer_prompt
    usr_txt = f"DOCUMENT: {source['doc_name']} -- p.{source['page_num']+1}"
    usr_txt += f"\n\nCONTENT:\n{source['content']}"
    usr_txt += f"\n\nQUERY: {query}"
    
    msg = [{"role": "system", "content": sys_txt}, {"role": "user", "content": usr_txt}]
    response = ''.join(get_gpt_response(msg))
    return source, response



def combine_answers(query, language, sub_answers):

    sys_txt = combine_answers_prompt

    src_txt = []
    for sub in sub_answers:
        temp = ""
        temp += f"DOC ID [{sub['doc_id']}]"
        temp += f"\nTITLE: {sub['doc_name']} -- p.{sub['page_num']+1}"
        temp += f"\nCONTENT:\n{sub['answer']}"
        src_txt.append(temp)
    src_txt = "\n\n---\n\n".join(src_txt)

    usr_txt = f"QUERY: {query}\nOUTPUT_LANGUAGE: {language}\n\nDOCUMENTS:\n\n{src_txt}"

    msg = [{"role": "system", "content": sys_txt}, {"role": "user", "content": usr_txt}]

    for token in get_gpt_response(msg):
        yield token


def format_output(eng_query, relevant_sources, language):
    if len(relevant_sources) == 0:
        yield "I don't know much about that"

    else:

        #add IDs to sources
        for source in relevant_sources:
            id = f"{source['doc_id']}.{source['page_num']}"
            source['id'] = id

        #make source payload
        source_payload = {"query":eng_query, "sources":[]}
        for x in relevant_sources:
            entry = {
                "episode_name": x['episode_name'],
                "show_name": x['show_name']
            }
            source_payload["sources"].append(entry)
        source_payload = json.dumps(source_payload, ensure_ascii=False)
        yield source_payload + "<EOM>"

        #synthesize answers
        token_generator = combine_answers(eng_query, language, relevant_sources)
        for token in token_generator:
            yield token

def search_relevant_sources(eng_query, filters, language):
    #perform source search
    sources = search_sources(eng_query, filters)
    print("SOURCES COUNT:", len(sources))
    for source in sources:
        print(source)



    #make document-wise answers
    #do this in batches of 10, and break whenever we have at least 1 hit
    relevant_sources = []
    with ThreadPoolExecutor(max_workers=10) as executor:

        for i in range(0, len(sources), 10):

            subset = sources[i:i+10]
            futures = []

            for source in subset:
                future = executor.submit(make_answer, source, eng_query)
                futures.append(future)

            for future in as_completed(futures):
                source, resp = future.result()
                if not "NONE" in resp:
                    source['answer'] = resp
                    relevant_sources.append(source)

            if len(relevant_sources) > 0:
                break

    return relevant_sources


def make_resp(query):

    relevant_sources = search_relevant_sources(query)
    print("RELEVANT SOURCES COUNT:", len(relevant_sources))
   

    format_output_generator = format_output(query, relevant_sources)
    for token in format_output_generator:
        yield token
    



