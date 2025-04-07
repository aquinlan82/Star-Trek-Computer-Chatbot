import { useRef, useEffect } from 'react';
import styled from '@emotion/styled'
import Message from './Message'

const OuterContainer = styled.div`
    height: 90%;
    width: 60%;
    display: flex;
    box-sizing: border-box;
    flex-direction: column;
    overflow-y: auto;
    scrollbar-gutter: stable;
    padding-inline: 20px;
    padding-block: 20px;
    &::-webkit-scrollbar {
        box-sizing: border-box;
        background: transparent;
        width: 7px;
    }
    &::-webkit-scrollbar-thumb {
        box-sizing: border-box;
        background: #414141;
        border-radius: 10px;
    }
    &::-webkit-scrollbar-track {
        box-sizing: border-box;
        background: #EEEEEE;
    }`

const InnerContainer = styled.div`
    display: flex;
    box-sizing: border-box;
    flex-direction: column;
    height: fit-content;`

const Messages = ({messages}) => {

    //whenever messages change, auto scroll to bottom
    const messagesEndRef = useRef(null);
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    return (
        <OuterContainer>
            <InnerContainer>
                {messages.map((msg, index) => (
                    //if index != 0 make margin top = 20px
                    <Message
                        style={index !== 0 ? {marginTop: "20px"} : {}}
                        key={index}
                        msgIndex={index}
                        msg={msg}
                        />
                ))}
                <div ref={messagesEndRef} style={{height: "0px"}}/>
            </InnerContainer>
        </OuterContainer>
    );
};

export default Messages;
