import styled from '@emotion/styled'
import { useState } from 'react'

const LoadSpan = styled.span`
    font-weight: bold;
    color: #231F20;
    font-size: 14px;
    letter-spacing: 5px;
    user-select: none;
    height: 30px;`

const LoadAnimation = () => {
    const [idx, setIdx] = useState(0)
    const progress = ['.', '..', '...','..', '.', '' ]
    setTimeout(() => {
        setIdx((idx + 1) % progress.length)
    }, 300)
    return (
        <LoadSpan>
            {progress[idx]}
        </LoadSpan>
    )
}

export default LoadAnimation