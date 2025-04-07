import styled from '@emotion/styled'
import Bar from './components/Bar'
import Messages from './components/Messages'
import Welcome from './components/Welcome'
import splashImage from './assets/bridge.png'


import { useMemo, useState, useRef, useEffect } from "react";

const Div = styled.div`
    display: flex; 
    height: 100%;
    width: 100%;
    
`

const PageContent = styled.div`
    display: flex;
    height: 100%;
    width: 100%;
    flex-direction: row;
    // background: linear-gradient(135deg, rgb(3,73,18), rgb(68,157,37));
    // background-color:rgb(240, 240, 240);
    // background: url("/background.png");
    justify-content: center;
    align-items: center;
    overflow: hidden;
    flex: 1;
    align-items: center;
`


const MainContainer = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    flex-grow: 1;
    gap: 20px;
    height: 100%;
    width: 100%;
    box-sizing: border-box;
    padding: 0;
    margin: 0;
    `

const Shade = styled.div`
    position: absolute;
    z-index: 2;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding-top: 20px;
    width: 100%;
    height: 100%;
    flex-shrink: 1;
    justify-content: space-between;
    background-color: rgba(0, 0, 0, 0.5);
`




function App() {

    const [messages, setMessages] = useState([]);
    const [inputEnabled, setInputEnabled] = useState(true);
 

    const processMsg = (text) => {
        
        //render user message and empty assistant message (loading)
        setInputEnabled(false);
        const newUserMsg  = {
            role: 'user',
            content: text,
            filters: {...selections},
            showLoading: false};
        const withUserMsg = [...messages, newUserMsg];
        const emptyBotMsg = {
            role: 'assistant',
            content: '', 
            query: '',
            sources: [],
            showLoading: true};
        const withEmptyMsg = [...withUserMsg, emptyBotMsg];
        setMessages(withEmptyMsg);

        //get server response for sources
        let states = ['searching','responding']
        let state = 0;
        let buffer = "";
        let query = "";
        let sources = [];
        let answer = "";
        let didTimeout = false;

        let body = JSON.stringify({"history":withUserMsg})

        // fetch('http://localhost:8000/chat', {signal: AbortSignal.timeout(90000)}, {
        fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: body
        })
        .then(response => { 

            console.log("start")

            const reader = response.body.getReader();

            return new Promise((resolve, reject) => {

                // Time out only applies to when the stream has begun but takes too long to respond
                // if the stream is not started, the server must send the timeout
                let timeout = null;

                function readNext() {
                    timeout = setTimeout(() => {
                        didTimeout = true;
                        reject(new Error("Timeout after 90 seconds"));
                    }, 90000);
                    return reader.read().then(processNext);
                }

                function processNext({ done, value }){ 

                    clearTimeout(timeout);

                    if (didTimeout) {
                        return;
                    }

                    if (done) {
                        resolve();
                        setInputEnabled(true);
                        return;
                    }

                    let rawText = new TextDecoder("utf-8").decode(value);
                    buffer += rawText;

                    //if searching, wait for complete message
                    if (states[state] === 'searching') {

                        //find index of "<EOM>" in buffer
                        let eomIndex = buffer.indexOf("<EOM>");

                        //EOM found, so we have a complete message
                        if (eomIndex !== -1) { 

                            //get part of buffer corresponding to message
                            let message = buffer.slice(0, eomIndex);
                            buffer = ""; //clear buffer

                            //now parse the message
                            message = JSON.parse(message);


                            //update variables
                            query = message.query;
                            sources = message.sources;
                            

                            //increment state
                            state++;

                        }

                        return readNext(); //continue reading

                    } else if (states[state] === 'responding') {
                        
                        answer = buffer;

                        let botMsg = {
                            role: 'assistant', 
                            content: answer,
                            query: query,
                            sources: sources,
                            showLoading: false};
                        let withBotMsg = [...withUserMsg, botMsg];
                        setMessages(withBotMsg);

                        return readNext();

                    }
                }

                //start the process
                readNext();

            });
        })
        .catch(error => {
            console.error(error)
            let chatRespMsg = {role: 'assistant', content: "Timeout Error", query: query, sources: sources, showLoading: false};
            let withChatResp = [...messages, newUserMsg, chatRespMsg];
            setMessages(withChatResp);
            setInputEnabled(true);


        });
    }


    //handle user message submission
    const processMsgDummy = (text) => {
        
        //render user message and empty assistant message (loading)
        setInputEnabled(false);
        const newUserMsg  = {
            role: 'user',
            content: text,
            showLoading: false};
        const withUserMsg = [...messages, newUserMsg]; 
        const botMsg = {
            role: 'assistant',
            content: 'This is a sample answer. ',
            query: text,
            showLoading: false};
        const withBotMsg = [...withUserMsg, botMsg];
        setMessages(withBotMsg);

    }



    
    return (

        <Div>
            <PageContent style={{ backgroundImage: `url(${splashImage})`, backgroundSize: 'cover', backgroundPosition: 'center', backgroundRepeat: 'no-repeat'}}>
                <MainContainer>
                    {/* <Shade/> */}

                    {/* <SplashImage src="/bridge.png"/> */}
                    {messages.length > 0 &&
                        <Messages messages={messages}/>
                    }

                    {messages.length === 0 &&
                        <Welcome processMsg={processMsgDummy}/>
                    }
                    <Bar
                        inputEnabled={inputEnabled}
                        processMsg={processMsgDummy}
                        />


                </MainContainer>
            </PageContent>
        </Div>
        


    );
}

export default App;
