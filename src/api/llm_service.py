import json
import os
from groq import Groq
from typing import List, Dict, Any, AsyncGenerator
from src.api.schemas import ChatMessage

# Initialize the Groq client using environment variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"

def build_system_prompt(search_results: List[Dict[str, Any]]) -> str:
    """Builds a system prompt injecting the retrieved search results as context."""
    context_str = json.dumps(search_results, indent=2)
    
    prompt = f"""You are a highly knowledgeable AI shopping assistant and search engine expert.
You help users find products, understand catalog items, and answer questions based ONLY on the provided Context.
If the answer is not in the Context, inform the user that you don't have that information.

Context (Top Search Results):
{context_str}

Respond conversationally, concisely, and professionally. When referencing items, use their exact titles."""
    return prompt

async def generate_chat_stream(messages: List[ChatMessage], search_results: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """Calls Groq API and yields the response stream."""
    
    # 1. Prepend the System Prompt to the messages
    system_prompt = build_system_prompt(search_results)
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    # 2. Call Groq
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=formatted_messages,
        temperature=0.7,
        max_completion_tokens=2048,
        top_p=1,
        stream=True
    )
    
    # 3. Yield chunks (Server-Sent Events format for FastAPI StreamingResponse)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content
