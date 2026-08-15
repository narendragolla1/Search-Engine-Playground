import json
from groq import Groq
from typing import List, Dict, Any, AsyncGenerator
from src.api.schemas import ChatMessage
from src.core.config import settings
from loguru import logger

if settings.groq_api_key == "your_groq_api_key_here":
    logger.warning("GROQ_API_KEY is not set in the .env file. AI Assistant will not work.")
    client = None
else:
    client = Groq(api_key=settings.groq_api_key)

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
    
    if client is None:
        yield "Error: GROQ_API_KEY is not configured in the .env file. Please add your API key to use the AI Assistant."
        return

    # 1. Prepend the System Prompt to the messages
    system_prompt = build_system_prompt(search_results)
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    # 2. Call Groq
    stream = client.chat.completions.create(
        model=settings.model_name,
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

async def analyze_schema(sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes a sample of JSON data and recommends schema settings for the search engine."""
    if client is None:
        raise ValueError("GROQ_API_KEY is not configured.")
        
    prompt = f"""You are a search engine architect. I am providing you with a sample of JSON data that I want to index.
Your task is to analyze the keys and values and recommend which fields should be indexed.

Sample Data:
{json.dumps(sample_data[:5], indent=2)}

Please return a JSON object with the following structure:
{{
    "searchable_fields": ["list of strings representing fields that should be used for full-text and semantic search (e.g. title, description, plot)"],
    "field_weights": {{"field_name": float_weight}},  // Give higher weights (e.g. 2.0 or 3.0) to important fields like titles.
    "filterable_fields": ["list of strings representing fields that should be used for exact filtering (e.g. category, year, author, price)"]
}}

Respond ONLY with valid JSON. Do not include markdown formatting or any other text."""

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    result_str = response.choices[0].message.content
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response as JSON: {result_str}")
        return {"searchable_fields": [], "field_weights": {}, "filterable_fields": []}

async def extract_intent(query: str, filterable_fields: List[str]) -> Dict[str, Any]:
    """Uses the LLM to extract structural filters from a natural language query."""
    if client is None or not filterable_fields:
        return {}
        
    prompt = f"""You are a search query analyzer. I will provide a user query and a list of valid filter fields.
Extract any mentioned filters from the query into a JSON object.

Valid Filter Fields: {filterable_fields}
Query: "{query}"

For example, if the valid fields are ["Category", "Color", "Size"] and the query is "cheap red running shoes size 10", you should output:
{{
    "Color": "red",
    "Category": "running shoes",
    "Size": "10"
}}

Respond ONLY with valid JSON containing the exact key names from the valid fields list. Do not include markdown or explanations. Return an empty JSON object {{}} if no filters are found."""

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result_str = response.choices[0].message.content
        return json.loads(result_str)
    except Exception as e:
        logger.error(f"Failed to extract intent: {str(e)}")
        return {}
