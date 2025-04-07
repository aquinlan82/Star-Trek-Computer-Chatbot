import styled from '@emotion/styled'

const Container = styled.div`
    display: flex;
    flex-direction: column;
    gap: 20px;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    flex-shrink: 1;
    text-align: center;
    color:rgb(189, 189, 189);
    user-select: none;
    font-family: Cabin, sans-serif;
    font-size: 60px;`

//questions should wrap to the next line if they exceed the width of the container
const Questions = styled.div`
    display: flex;
    flex-direction: row;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    min-height: 110px;`

const Question = styled.div`
    background-color: white;
    border-radius: 10px;
    box-sizing: border-box;
    color: #3a3a3a;
    padding: 10px;
    height: 100px;
    width: 190px;
    font-size: 16px;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    box-shadow: 0px 0px 3px 1px rgba(0, 0, 0, 0.3);
    &:hover {
        box-shadow: 0px 0px 6px 1px rgba(0, 0, 0, 0.3);
    }
    &:active {
        box-shadow: 0px 0px 1px 1px rgba(0, 0, 0, 0.3);
    }`

const Welcome = ({processMsg}) => {

    return (
        <Container>
        </Container>
    );
}

export default Welcome;
