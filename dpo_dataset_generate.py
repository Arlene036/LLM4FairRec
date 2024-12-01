import pandas as pd
import asyncio
import random
from openai import AsyncOpenAI
from datasets import Dataset, DatasetDict
import re
import os
from typing import List
# OpenAI Async Dispatch
async def generate_chosen_reasons(user_histories, chosen_movies_list, model="gpt-4o-mini"):
    """
    Generate reasons for chosen reranking of movie recommendations.
    """
    prompts = []
    for user_history, chosen_movies in zip(user_histories, chosen_movies_list):
        chosen_request = "\n".join(
            [
                f"Rank{i + 1}: {m}"
                for i, m in enumerate(chosen_movies)
            ]
        )

        prompt = f"""
        You are a AI assistant to generate short reasons for specific reason of ranked movies, why the movies should be ranked like this.

        User's movie watching history: {user_history}
     
        -Reranked movies:
        {chosen_request}

        Explain why the following movies are ranked in their respective positions and match the user's preferences,one movie in a line.
        Possible responses for chosen reranking reason could be:
        This movie combines science fiction, thriller, and mystery elements, aligning strongly with the user's preference for complex, thought-provoking stories as seen in their history.
        The mathematical genius theme and psychological intensity resonate with the user's interest in thrilling and intelligent narratives.
        The movie tells a romantic story happening in high school which is similar to user's interested movie.

        Please STRICTLY follow the format below for output, the ranked order should be the same as the 'Reranked movies' given, don't output any other words, do not repeatedly output one movie, just one movie a line with the reason, do not output all the movie information except the id
        Rank 1: {chosen_movies[0].split(':')[0]} - Reason: (the reason that theuser might like this movie most)
        Rank 2: {chosen_movies[1].split(':')[0]} - Reason: ...
        ...
        Rank {len(chosen_movies)-1}: {chosen_movies[-2].split(':')[0]}  - Reason: (Reason that the user possible not like this movie)
        Rank {len(chosen_movies)}: {chosen_movies[-1].split(':')[0]}  - Reason: (Reason that the user possible not like this movie)
        """
        prompts.append(prompt)
    

    responses = await dispatch_openai_requests(
        prompts, model=model, temperature=0.4, max_tokens=500
    )
    
    all_reasons = []
    for response in responses:
        if response:
            reasons = response.replace("Chosen:", "").strip().split("\n")
            all_reasons.append(reasons)
        else:
            all_reasons.append([])
    
    return all_reasons

async def dispatch_openai_requests(
    messages_list: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[str]:
    """
    Dispatch OpenAI API requests asynchronously.
    """
    async with AsyncOpenAI(
        api_key="",
        base_url="https://cmu.litellm.ai",
    ) as client:
        async def single_request(text: str) -> str:
            try:
                if 'gpt' in model:
                    # For GPT-based models
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
                    # For non-chat models
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

        # Gather all tasks asynchronously
        tasks = [single_request(text) for text in messages_list]
        return await asyncio.gather(*tasks)

# Generate Prompt
def generate_prompt(history, candidates, len_candidates):
    return f"""You are a recommender system. Based on a user's historical likes and dislikes, rank the given candidate movies by their likelihood of being the user's next favorite, according to their watching history. Please think step by step.

This user's historical interactions: {history}
There are {len_candidates} Candidates for recommendation: {candidates}

Strictly follow the output format:
Rank1: movieid - Reason: explain why the user would most likely enjoy this movie
Rank2: movieid - Reason: explain why the user would likely enjoy this movie second
...
Rank{len_candidates}: movieid - Reason: explain why the user would like this movie last
Please provide a ranked list of the recommended movies. You MUST rank only the given candidates and cannot include any movies not listed in the candidate list."""

import re

def parse_movie_list(movie_string):
    """
    Parses a string of movies into a list of movie descriptions.
    """
    # Split by "Movie" to separate each movie entry
    movies = re.split(r'Movie (\d+):', movie_string)
    parsed_movies = []
    parsed_movies_id = []

    # Process the split results to extract movie details
    for i in range(1, len(movies), 2):  # Skip the first split as it's before "Movie"
        movie_id = movies[i].strip()  # Extract movie ID
        movie_details = movies[i + 1].strip()  # Extract details
        if movie_details.endswith(','):
            movie_details = movie_details[:-1]
        parsed_movies.append(f"Movie {movie_id}:{movie_details}")
        parsed_movies_id.append(movie_id)
    return parsed_movies, parsed_movies_id

async def process_row(rows):
    """
    批量处理多行数据
    """
    user_histories = []
    chosen_movies_list = []
    rejected_movies_list = []
    prompts = []
    
    for row in rows:
        history, history_id = parse_movie_list(row['history'])
        candidates, candidates_id = parse_movie_list(row['candidates'])
        ground_truth, ground_truth_id = parse_movie_list(row['ground_truth'])
        
        user_history_desc = " ".join(history)
        all_movies = candidates

        # 准备chosen和rejected列表
        ground_truth_movies = [movie for movie, movie_id in zip(candidates, candidates_id) 
                             if movie_id in ground_truth_id]
        non_ground_truth_movies = [movie for movie, movie_id in zip(candidates, candidates_id) 
                                 if movie_id not in ground_truth_id]

        random.shuffle(ground_truth_movies)
        random.shuffle(non_ground_truth_movies)
        chosen_movies = ground_truth_movies + non_ground_truth_movies
        
        rejected_movies = all_movies[:]
        random.shuffle(rejected_movies)
        
        # 收集数据
        user_histories.append(user_history_desc)
        chosen_movies_list.append(chosen_movies)
        rejected_movies_list.append(rejected_movies)
        
        # 生成prompt
        prompt_candidates = all_movies[:]
        random.shuffle(prompt_candidates)
        prompts.append(generate_prompt(
            user_history_desc,
            ", ".join(prompt_candidates),
            len(prompt_candidates)
        ))
    
    # 批量生成原因
    chosen_reasons = await generate_chosen_reasons(user_histories, chosen_movies_list)
    rejected_reasons = await generate_chosen_reasons(user_histories, rejected_movies_list)
    
    return [
        {
            "prompt": prompt,
            "chosen": "\n".join(chosen_reason),
            "rejected": "\n".join(rejected_reason)
        }
        for prompt, chosen_reason, rejected_reason 
        in zip(prompts, chosen_reasons, rejected_reasons)
    ]

# Process Dataset Incrementally
async def process_dataset_incrementally(input_path, output_path, dataset_name, batch_size=1000):
    dataset = pd.read_csv(input_path)
    processed_rows = []
    existed_length = 0
    
    if os.path.exists(output_path):
        existed_dataset = pd.read_csv(output_path)
        existed_length = len(existed_dataset)
        print(f"Found existing dataset with {existed_length} rows")
    
    remaining_dataset = dataset.iloc[existed_length:]
    total_remaining = len(remaining_dataset)
    print(f"Processing remaining {total_remaining} rows")
    
    for i in range(0, total_remaining, batch_size):
        batch = remaining_dataset.iloc[i : i + batch_size]
        # 将batch转换为字典列表
        batch_records = batch.to_dict('records')
        results = await process_row(batch_records)  # 现在传递字典列表
        
        processed_rows.extend(results)
        if existed_length == 0:
            pd.DataFrame(processed_rows).to_csv(output_path, index=False)
        else:
            pd.DataFrame(processed_rows).to_csv(output_path, index=False, mode='a', header=False)
        
        print(f"Processed batch {i//batch_size + 1}, total processed: {len(processed_rows)}")
        if i % 6000 == 0:
            upload_to_huggingface(output_path, dataset_name)

# Upload to Hugging Face
def upload_to_huggingface(output_path, dataset_name):
    processed_df = pd.read_csv(output_path)
    hf_dataset = Dataset.from_pandas(processed_df)
    DatasetDict({"train": hf_dataset}).push_to_hub(dataset_name)

import asyncio

# Define your main async function
async def main():
    # Call your async function here
    input_csv = 'dataset/movielens_1M/natural_language_top10.csv'
    output_csv = 'dataset/movielens_1M/dpo_dataset_3.csv'
    dataset_name = "sssssssshhhhhu/movielens_dpo_dataset_3"
 
    # Call the processing function
    await process_dataset_incrementally(input_csv, output_csv, dataset_name, batch_size=100)

    # Call the Hugging Face upload function
    upload_to_huggingface(output_csv, dataset_name)

# Run the async function using asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())
    #python dpo_dataset_generate.py
    #huggingface-cli login
    