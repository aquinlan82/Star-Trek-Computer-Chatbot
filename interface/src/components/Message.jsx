import styled from '@emotion/styled'
import LoadAnimation from './LoadAnimation'
import ReactMarkdown from 'react-markdown';
import "../../styles.css"
import { useEffect, useState } from 'react'

const Container = styled.div`
    display: flex;
    line-height: 25px;
    flex-direction: column;
    width: 100%;
    padding-inline: 10px;
    padding-block: 8px;
    min-height: 30px;
    max-width: 100%;
    min-width: 40px;
    width: fit-content;
    justify-content: center;
    align-items: center;
    font-family: Cabin, sans-serif;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 400;
    word-wrap: break-word;
    box-shadow: 0px 0px 3px 1px rgba(0, 0, 0, 0.2);
    overflow-wrap: anywhere;
    white-space: pre-wrap;`

const UserContainer = styled(Container)`
    margin-left: 20px;
    background-color: rgb(221, 158, 147);
    max-width: 70%;
    align-self: flex-end;
    color: black;
    border-top-right-radius: 0px;`

const BotContainer = styled(Container)`
    background-color: rgb(153, 147, 221);
    align-self: flex-start;
    max-width: 70%;
    color: #231F20;
    border-top-left-radius:0px;`

const Link = styled.button`
    color: #EEEEEE;
    text-align: center;
    padding-block: 2px;
    padding-inline: 4px;
    margin-inline: 1px;
    font-weight: bold;
    font-size: 12px;
    font-family: sans-serif;
    white-space: nowrap;
    border-radius: 3px;
    text-decoration: none;
    background-color: rgb(0, 94, 29);
    border: none;
    user-select: none;
    cursor: pointer;
    &:hover {
        background-color: rgb(1, 126, 41);
    }
    &:active {
        background-color: rgb(1, 52, 17);
    }`


const SelectionContainer = styled.div`
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
`

const SelectionPill = styled.div`
    display: flex;
    border: 2px solid rgb(0, 94, 29);
    border-radius: 20px;
    padding-inline: 10px;
    margin-right: 10px;
    margin-bottom: 10px;
    color: black;

`

const NoticeDiv = styled.div`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
`

 

const Message = ({msgIndex, msg, ...props}) => {
    
    const handleLinkClick = (e, sourceIdx) => {
        let source = msg.sources[sourceIdx];
        window.open(source.link, '_blank');
        e.stopPropagation();
    }

    console.log("Message: ", msg)

    if (msg.role == 'user') {
        return (
            <UserContainer style={props.style}>
                {msg.content}
            </UserContainer>
        )
    } else if (msg.role == 'assistant') {

        var lines = msg.content.split("CITATIONS:");
        if (lines.length == 1) {
            lines = [lines[0], ""]
        } 
            
        // get text before the first citation and remove the citations and pre period spaces
        var text = lines[0].replace(/(\[[^\]]+\])/g, '').replace(/\s*\./g, '\.')

        // get the citations and remove the spaces
        var citations = lines[1].split(/(\[[^\]]+\])/);
        citations = citations.filter(line => line.trim() !== '' && line.trim() !== '.' && line.trim() !== 'CITATIONS:')

        return (

            <BotContainer style={props.style}>   
                { !msg.showLoading &&
                    <div>

                        <div className='markdown-container'>
                            <ReactMarkdown>{text}</ReactMarkdown>
                        </div>
                        {
                        citations.map(function (line, index) {
                            let citation_number = parseInt(line.substring(1,line.length-1))
                            let source_index = 0;
                            msg.sources.map((source, index) => {
                                if (source.id == citation_number) {
                                    source_index = index
                                }
                            })
                            
                            return (
                                <Link key={index} onClick={(e) => {handleLinkClick(e, source_index)}}>
                                    {msg.sources[source_index].doc_name}
                                </Link>)
                        }
                            
                        )}

                    </div>
            }
                
                {msg.showLoading ? <LoadAnimation/> : null}

            </BotContainer>
        )
    }
}

export default Message
