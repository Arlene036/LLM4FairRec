import os
import asyncio
from openai import AsyncOpenAI

async def dispatch_openai_requests(
    messages_list: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    async with AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://cmu.litellm.ai",
    ) as client:
        async def single_request(text: str) -> str:
            try:
                if 'gpt' in model:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result = response.choices[0].message.content
                else:
                    response = await client.completions.create(
                        model=model,
                        prompt=text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result = response.choices[0].text
                return result
            except Exception as e:
                print(f"Error in request: {str(e)}")
                return f"Error: {str(e)}"

        tasks = [single_request(text) for text in messages_list]
        return await asyncio.gather(*tasks)