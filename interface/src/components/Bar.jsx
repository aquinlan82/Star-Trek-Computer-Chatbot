import styled from '@emotion/styled'
import SubmitIcon from '../assets/icons/submit_icon.svg?react'
import ArrowIcon from '../assets/icons/chevron_right.svg?react'

import { useState, useRef, useEffect } from 'react';


const Container = styled.div`
    position: sticky;
    bottom: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;
    flex-grow: 0;
    width: 60%;
    margin: 15px;
    `

const Backer = styled.div`
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    border-radius: 50px;
    background: rgba(255, 255, 255, 0.975);
    min-height: 60px;
    padding-inline: 10px;
    padding-block: 14px;
    margin-bottom: 20px; 
    box-shadow: 0px 0px 3px 1px rgba(0, 0, 0, 0.2);
    padding-inline: 20px;
    width: 100%;`

const TextInput = styled.textarea`
    border: none;
    resize: none;
    padding: 0px;
    background: none;
    flex-grow: 1;
    margin-inline: 10px;
    color: #231F20;
    font-size: 18px;
    font-weight: 400;
    line-height: normal;
    font-family: Cabin, sans-serif;
    &:disabled {
        color: #8D8989;
        cursor: default;
    }
    &:focus {
        outline: none;
    }
    &::-webkit-scrollbar {
        background: transparent;
        width: 7px;
    }
    &::-webkit-scrollbar-thumb {
        background: #414141;
        border-radius: 10px;
    }`

const Buttons = styled.div`
    display: flex;
    flex-direction: row;
    width: 40px;
    height: 100%;
    justify-content: center;
    align-items: center;` 

const SubmitButton = styled(SubmitIcon)`
    cursor: pointer;
    height: 24px;
    width: 24px;`

const Arrow = styled(ArrowIcon)`
    height: 24px;
    width: 24px;`

const NewButton = styled.button`
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #4c4c4c;
    color: white;
    cursor: pointer;
    font-size: 14px;
    border-radius: 30px;
    width: 100px;
    height: 30px;
    border: none;

    &:hover {
        border: none;
        background-color: #333333;
    }
`

const Notice = styled.div`
    align-items: center;
    width: min(90%, 800px);
    color: #404040;
    font-size: 15px;
    font-family: Cabin, sans-serif;
    font-weight: 400;
    user-select: none;
    text-align: center;`



const Bar = ({inputEnabled, processMsg, openNew}) => {

    const [hasText, setHasText] = useState(false); //track if the text input has text
    const textInputRef = useRef(null);  // reference to TextInput

    //triggered on Submit button click or Enter key press
    const handleSubmit = () => {

        if (hasText) {
            const textInput = textInputRef.current;
            processMsg(textInput.value.trim())
            textInput.value = "";
            sizeBar();
            setHasText(false);

        }
    }

    //resize the text input box to fit the text
    const sizeBar = () => {
        const textInput = textInputRef.current;
        textInput.style.height = '0px';
        let newHeight = textInput.scrollHeight;
        newHeight = Math.ceil(newHeight);
        newHeight = Math.min(newHeight, 200);
        textInput.style.height = newHeight + 'px';
    }

    //triggered on Enter key press
    const handleKeyPress = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    }

    //triggered on TextInput value change
    const handleChange = (e) => {
        const text = e.target.value;
        setHasText(text.trim().length > 0);
        sizeBar();
    }
        
    return (
        <Container>

            <Backer>
                


                <TextInput
                    ref={textInputRef}
                    placeholder="Type your message here..."
                    disabled={!inputEnabled}
                    style={{cursor: inputEnabled ? 'text' : 'default'}}
                    onKeyDown={handleKeyPress}
                    onChange={handleChange}
                    rows={1}/>

                <Buttons>
                    <SubmitButton
                        style={{
                            "fill" : hasText ? '#C41230' : '#8D8989',
                            "cursor" : hasText ? 'pointer' : 'default'}}
                        alt="Submit Button"
                        onClick={handleSubmit}/>
                </Buttons>

            </Backer>

        </Container>
    );
  }
  
  export default Bar;
