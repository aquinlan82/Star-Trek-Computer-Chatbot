# basic imports
import os, tiktoken, base64, boto3, botocore, pickle, unicodedata, re, datetime
from pathlib import Path
from io import BytesIO
from scipy.sparse import dok_matrix
import pandas as pd
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

# openai and threading imports
from openai import AzureOpenAI
from openai._types import NOT_GIVEN
from openai._exceptions import RateLimitError
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# api imports
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# dotenv for environment variables
from dotenv import load_dotenv
load_dotenv()


"""
Check if a path exists, and create it if it doesn't.
Input: path (str)
"""
def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pickle_wrapper(path, method, data=None):
    if method == "read":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif method == "write":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Method must be 'read' or 'write'")


"""
Set up gpt client for Azure OpenAI.
Input: model name (str)
Output: gpt client (AzureOpenAI object)
"""
def get_gpt_client(model_name):
    #load GPT tokenizer
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")

    # possible clients
    gpt35_client = AzureOpenAI(
        api_key=os.environ["OPENAI_35_KEY"],
        api_version="2024-02-15-preview",   
        azure_endpoint="https://agco-can-sandbox.openai.azure.com/")

    gpt4o_client = AzureOpenAI(
            api_key=os.environ["OPENAI_4o_KEY"],
            api_version="2024-02-15-preview",
            azure_endpoint="https://agco-sandbox.openai.azure.com/")
    

    if model_name == "gpt-35-turbo":
        return gpt35_client
    elif model_name == "gpt-4o":
        return gpt4o_client
    else:
        raise ValueError("Invalid model name. Choose 'gpt-35-turbo' or 'gpt-4o'.")

"""   
Given the filenames of images, create a list of messages containing the images for gpt4o
Input: list of strings containing filenames
Output: list of dictionaries representing the image messages
"""
def make_image_messages(filenames):
    messages = []
    for filename in filenames:
        with open(filename, "rb") as image_file:
            image = image_file.read()
            img = base64.b64encode(image).decode('utf-8')
            
        img_msg = {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}",
                },
                },
            ],
        }
        messages.append(img_msg)
    return messages

 


"""
Send a message to the GPT model and return the response (non streaming)
Input: list of dictionaries representing the messages, gpt client object
Output: string containing the response
"""
@retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
def get_gpt_response(messages, client, model="gpt4o", force_json=False, temp=0.5, seed=42, max_tok=4096):
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


"""
List files in a directory with optional filtering by extension and recursion.

Args:
    path (str): The path to the directory.
    dirs (bool): Whether to include directories in the listing.
    files (bool): Whether to include files in the listing.
    extension (str): The file extension to filter by (e.g., '.txt').
    recursive (bool): Whether to search recursively in subdirectories.
    full_path (bool): Whether to return full paths or just names.

Returns:
    list: A list of file and/or directory paths.
"""
def list_files(path, include_dirs=True, include_files=True, include_extension=None, recursive=False, full_path=False, as_paths=False):

    items = []
    for root, dirs, files in os.walk(path):
        if include_dirs and not full_path:
            items.extend([d for d in dirs if os.path.isdir(os.path.join(root, d))])
        if include_dirs and full_path:
            items.extend([os.path.join(root, d) for d in dirs if os.path.isdir(os.path.join(root, d))])
        if include_files and not full_path:
            items.extend([f for f in files if os.path.isfile(os.path.join(root, f)) and (include_extension is None or f.endswith(include_extension))])
        if include_files and full_path:
            items.extend([os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f)) and (include_extension is None or f.endswith(include_extension))])
            
        if not recursive:
            break

    items = [x.replace("\\", "/") for x in items]  # Normalize paths to use forward slashes
    if as_paths:
        items = [Path(x) for x in items]
    return items


""""
Get the list of files in a directory and return a DataFrame with the file names and their corresponding paths."
"""
def get_aws_client():
    aws_session = boto3.Session()
    s3_client = aws_session.client(
        's3', 
        config = botocore.client.Config(
            s3={'use_accelerate_endpoint': True},
            max_pool_connections=10000)
    )
    aws_account_num = boto3.client('sts').get_caller_identity().get('Account')
    S3_BUCKET = f"sales-and-marketing-{aws_account_num}"

    return s3_client, S3_BUCKET

"""
Save a local file to an S3 bucket.
Input: s3 client object, bucket name (str), file path(str), object name (str)
"""
def save_s3_file(s3_client, bucket_name, file_path, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File {file_path} uploaded to {object_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Error uploading file {file_path}: {e}")

"""
Save a code variable to an S3 bucket.
Input: s3 client object, bucket name (str), object name (str), data (any)
"""
def save_s3_object(s3_client, bucket_name, object_name, data):
    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=data)
        print(f"Object {object_name} uploaded to bucket {bucket_name}.")
    except Exception as e:
        print(f"Error uploading object {object_name}: {e}")

"""
Download a file from an S3 bucket to a local file.
Input: s3 client object, bucket name (str), object name (str), file path (str)
"""
def download_s3_asfile(s3_client, bucket_name, object_name, file_path):
    try:
        s3_client.download_file(bucket_name, object_name, file_path)
        print(f"Object {object_name} downloaded to {file_path}.")
    except Exception as e:
        print(f"Error downloading object {object_name}: {e}")

"""
Download an object from S3 and return it as a byte string.
Input: s3 client object, bucket name (str), object name (str)
Output: byte string of the object data
"""
def download_s3_asobject(s3_client, bucket_name, object_name):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        data = response['Body'].read()
        print(f"Object {object_name} downloaded.")
        return data
    except Exception as e:
        print(f"Error downloading object {object_name}: {e}")
        return None
    


"""
Build a JSON-like dictionary from a product hierarchy DataFrame.
Input: include_alls (bool), filters (dict)
         filters: dictionary of column names and values to include    {"series": ["series1", "series2"], "model": ["model1"]}
Output: dictionary representing the product hierarchy
"""
def build_product_hierarchy_json(include_alls=False, filters=None):
    df = pd.read_pickle("product_hierarchy.pkl")

    # Filter the DataFrame based on the filters provided
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]


    # Organize the DataFrame into a nested dictionary structure
    if include_alls == False:
        out = {}
        for category in df['category_name'].unique():
            out[category] = {}
            for series in df[df['category_name'] == category]['series_code'].unique():
                out[category][series] = df[(df['category_name'] == category) & (df['series_code'] == series)]['model_code'].unique().tolist()

    else:
        out = {}
        for category in df['category_name'].unique():
            out["All Product Groups"] = {}
            out["All Product Groups"]["All Series"] = df['model_code'].unique().tolist()
            out["All Product Groups"]["All Series"].insert(0, "All Models")

            for series in df['series_code'].unique():
                out["All Product Groups"][series] = df[(df['category_name'] == category) & (df['series_code'] == series)]['model_code'].unique().tolist()
                out["All Product Groups"][series].insert(0, "All Models")




            out[category] = {}
            out[category]["All Series"] = df[(df['category_name'] == category)]['model_code'].unique().tolist()
            out[category]["All Series"].insert(0, "All Models")

            for series in df[df['category_name'] == category]['series_code'].unique():
                out[category][series] = df[(df['category_name'] == category) & (df['series_code'] == series)]['model_code'].unique().tolist()
                out[category][series].insert(0, "All Models")

                
    return out

def create_feedback_template():
    data = {"query": "Test question", 
        "filters": str({"group": "All Groups", "model": "All Models", "series": "All Series"}), 
        "rating": "Very Helpful",
        "comment": "This is a test comment",
        "response": "This is a test response",
        "sources": str([{"doc_name": "Test Document", "page_num": 1, "doc_id": 1}])
    }

    df = pd.DataFrame({"rating":data['rating'], "comment":data['comment'], "query":data['query'], "response":data['response'], "filters":data['filters'], "sources": data["sources"], "time": str(datetime.datetime.now())}, index=[0])
    return df


""" 
This function reads in the file at pdf_filename and returns a list
Each entry in the list represents a page and contains the image of it in byte format
Files are saved in format originalname_pagenum.png in the img directory provided
seems to crash pdfium to use threads
"""
def pdf2images(pdf_filename:Path, img_dir):
    imgs = [] 

    # create pdf object for document
    pdf = pdfium.PdfDocument(str(pdf_filename))
    for idx in range(len(pdf)):
        page = pdf[idx]
        #get contents from page
        bitmap = page.render(scale=2)
        
        #make img bytes as png file
        img = bitmap.to_pil()
        img_fp = BytesIO()
        img.save(img_fp, format='png')
        img_bytes = img_fp.getvalue()

        #make entry
        imgs.append(img_bytes)

    # save images
    for idx, img in enumerate(imgs):
        with open(f"{img_dir}/{pdf_filename.stem}_{idx}.png", "wb") as f:
            f.write(img)


    return len(imgs)


"""
Converts PIL Image object to a base64 string
the string can be added to a gpt call for analysis
"""
def make_img_url(img_obj): 
    image_file = BytesIO()
    img_obj.save(image_file, format="JPEG")
    byte_str = base64.b64encode(image_file.getvalue()).decode("utf-8")
    url = f"data:image/jpeg;base64,{byte_str}"
    return url




def do_ocr(img_filename, client):

    # Change prompt based on specifics of what should be extracted from document
    prompt = """You are a helpful assistant. Your job is to extract detailed text from an pdf about AGCO agriculture company.
    Ensure that every sentence is complete, as informative as possible (with numbers, brands, models, and all other details mentioned),
    and translated to English. Do NOT have any fragments, bullet points, or numbered lists. Describe all images as thoroughly as possible, 
    including names and brands. The length of what you extract should be similar to the length of the text in the original document."""

    # Create a prompt to send to the API
    sys_prompt = {
        "role": "system",
        "content": prompt
    }

    # Format the input to include an image
    img = Image.open(img_filename)
    img_msg = {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": make_img_url(img),
            },
            },
        ],
        }
    messages = [sys_prompt, img_msg]

    return get_gpt_response(messages, client, model="gpt-4o")



            
def thread_wrapper(max_workers, function, input_list):
    with tqdm(total=len(input_list)) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in input_list:
                future = executor.submit(function, *i)
                futures[future] = i

            for future in as_completed(futures):
                try:
                    result = future.result()
                    i = futures[future]

                    pbar.update(1)
                    yield [result, i]
                except Exception as e:
                    print(f"Error: {e}")


def get_text_dict(txt_directory:str)->dict:
    texts = {}
    for filename in os.listdir(txt_directory):
        file_label = filename.split(".")[0]
        try:
            with open(os.path.join(txt_directory, filename), 'r') as file:
                texts[file_label] = file.read()
        except: 
            with open(os.path.join(txt_directory, filename), 'r', encoding='utf8') as file:
                texts[file_label] = file.read()
    return texts

# split on new line and do groups of 3
# assumes pdf texts are named as pdfname_pagenumber.txt
# TODO allow overlap and more complicated ways of splitting sentences
def chunk_text(texts:dict, metadata:dict):

    chunks = []
    vectors = []

    CHUNK_SIZE = 3
    for filename in texts:
        text = texts[filename]
        pdf_name, page_num = filename.rsplit('_', 1)
        data = {'doc_name': pdf_name, 'page_num': page_num, 'doc_id': metadata[pdf_name]['id']}
        sentences = re.split('\n', text)
        sentences = [sentence for sentence in sentences if sentence != '']
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = '\n'.join(sentences[i:i+CHUNK_SIZE])
            chunks.append(chunk)
            vectors.append(data)

    return chunks, vectors


def get_vectorizer_variables():
    vectorizer = SentenceTransformer('BAAI/bge-base-en-v1.5')
    ngram2idx = pickle.load(open( '../data/ngram2idx.bin', 'rb'))
    stop_eng = set(open('../data/stop_english.txt').read().splitlines())
    return vectorizer, ngram2idx, stop_eng



def clean_text(text:str)->str:
    
    #normalize text to NFKC form, ie replace composed characters with decomposed characters
    text = unicodedata.normalize('NFKC', text)
    
    #normalize case
    text = text.lower()

    #map common accents to ascii
    accent_map = {
        # Acute accents
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        # Grave accents
        "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
        # Umlauts
        "ä": "a", "ë": "e", "ï": "i", "ö": "o", "ü": "u",
        # Circumflex
        "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u",
        # Cedilla
        "ç": "c",
        # Additional Portuguese characters
        "ã": "a", "õ": "o",
        # Additional Spanish characters
        "ñ": "n",
        # Additional German characters
        "ß": "ss",
    }

    for accent, replacement in accent_map.items():
        text = text.replace(accent, replacement)

    #remove html tags
    html_stop = ["<td>","</td>","<tr>","</tr>","<th>","</th>","<table>","</table>","<tbody>","</tbody>"]
    for tag in html_stop:
        text = text.replace(tag, "\n")

    #remove sequences of 1+ "#" characters
    text = re.sub(r'#+', '', text)

    #replace multple \n with only one
    text = re.sub(r'\n+', '\n', text)

    return text


def get_ngrams(text:str, stop_eng)->set:

    output = set()

    #clean text formatting
    text = clean_text(text)

    #split text into chunks on whitespace
    chunks = set()
    temp = set(re.split(r'\s+', text))
    for word in temp:
        
        #remove any leading/trailing non-word characters
        word = re.sub(r'^[\W_]+', '', word, flags=re.UNICODE)
        word = re.sub(r'[\W_]+$', '', word, flags=re.UNICODE)
        
        if word in stop_eng:
            continue
        
        if len(word) < 2:
            continue
            
        #check if word has both digits and non-digit characters (useful for part nums / fault codes)
        if re.search(r'\d', word, flags=re.UNICODE) and\
            re.search(r'\D', word, flags=re.UNICODE):
                
                #add entire word to output
                output.add(word)
                
                #split word on non-word chars and add to chunks
                temp = re.split(r'[\W_]+', word, flags=re.UNICODE)
                output.update(temp)
        
        else:
            #add word to be split into ngrams
            chunks.add(word)

    #create ngrams
    for chunk in chunks:

        N = 4
        has_chinese = re.search(r'[\u4e00-\u9fff]', chunk)
        if has_chinese:
            N = 2 #chinese has more info per character
        
        padded_chunk = f" {chunk} "
        for i in range(max(1,len(padded_chunk)-N+1)):
            output.add(padded_chunk[i:i+N])

    return output

def make_lexical_vector(text:str, vectorizer_vars)->dok_matrix:
    vectorizer, ngram2idx, stop_eng = vectorizer_vars
    text = clean_text(text)
    ngrams = get_ngrams(text, stop_eng)
    vector = dok_matrix((1,len(ngram2idx)))
    indices = [ngram2idx[ngram] for ngram in ngrams if ngram in ngram2idx]
    vector[0,indices] = 1
    return vector



def make_semantic_vector(text:str, vectorizer_vars)->list:
    vectorizer, ngram2idx, stop_eng = vectorizer_vars
    vector = vectorizer.encode(text, show_progress_bar=False)
    return vector


def make_vector_wrapper(vectors, chunks, make_function, vectorizer_vars):
    
    failures = []
    output_vectors = []
    with tqdm(total=len(vectors)) as pbar:
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {}
            for vector_data, chunk in zip(vectors, chunks):
                future = executor.submit(make_function, chunk, vectorizer_vars)
                futures[future] = vector_data.copy()
            for future in as_completed(futures):
                try:
                    out = future.result()
                    vector_data = futures[future]
                    vector_data['vector'] = out
                    output_vectors.append(vector_data)
                    
                except Exception as e:
                    print(f"Error [{vector_data}]: {e}")
                    failures.append(futures[future])
                    exit()
                pbar.update(1) 

    return output_vectors, failures




def get_zilliz_client():
    client = MilvusClient(
        uri="https://in05-2ecc42d8837bc9a.serverless.gcp-us-west1.cloud.zilliz.com",
        token=os.getenv('MILVUS_API_KEY')
    )
    return client



def save_to_zilliz(collection_name, filename_list, zilliz_client, batch_size=1000):
    for filename in filename_list:
        vectors = pickle.load(open(filename), "rb")

        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            zilliz_client.insert(collection_name=collection_name, data=batch_vectors)
