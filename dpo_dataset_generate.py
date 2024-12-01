import pandas as pd
import asyncio
import random
from openai import AsyncOpenAI
from datasets import Dataset, DatasetDict
import re
import os
from typing import List
# OpenAI Async Dispatch
async def generate_chosen_reasons(user_history, chosen_movies, model="gpt-4o-mini"):
    
    """
    Generate reasons for chosen reranking of movie recommendations.
    """
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
    #print(prompt)
    # Call OpenAI and handle response
    # print(f"Calling OpenAI with prompt: {prompt}")
    # print('--------------------------------')
    # print(chosen_request)
    response = await dispatch_openai_requests(
        [prompt], model=model, temperature=0.4, max_tokens=500
    )
  
    if response and len(response) > 0:
        reasons = response[0].replace("Chosen:", "").strip().split("\n")
        return reasons

    return []



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

# # Generate Chosen and Rejected
# async def generate_chosen_and_rejected(ground_truth, candidates, openai_model="gpt-4"):
#     # Sort chosen (ground truth first, then remaining candidates)
#     chosen_order = ground_truth + [c for c in candidates if c not in ground_truth]
#     chosen_reasons = await dispatch_openai_requests(
#         [f"Why would the user like movie {c} the most?" for c in chosen_order],
#         model=openai_model,
#     )
#     chosen = [f"Rank{i + 1}: {chosen_order[i]} - Reason: {chosen_reasons[i]}" for i in range(len(chosen_order))]

#     # Shuffle rejected candidates
#     rejected_candidates = [c for c in candidates if c not in ground_truth]
#     random.shuffle(rejected_candidates)
#     rejected_reasons = await dispatch_openai_requests(
#         [f"Why would the user not prefer movie {c}?" for c in rejected_candidates],
#         model=openai_model,
#     )
#     rejected = [f"Rank{i + 1}: {rejected_candidates[i]} - Reason: {rejected_reasons[i]}" for i in range(len(rejected_candidates))]

#     return chosen, rejected

# Process Single Row

import re

def parse_movie_list(movie_string):
    """
    Parses a string of movies into a list of movie descriptions.
    """
    # Split by "Movie" to separate each movie entry
    movies = re.split(r'Movie (\d+):', movie_string)
    parsed_movies = []

    # Process the split results to extract movie details
    for i in range(1, len(movies), 2):  # Skip the first split as it's before "Movie"
        movie_id = movies[i].strip()  # Extract movie ID
        movie_details = movies[i + 1].strip()  # Extract details
        parsed_movies.append(f"Movie {movie_id}:{movie_details}")

    return parsed_movies


async def process_row(row):
    # Parse the history, candidates, and ground_truth fields
    history = parse_movie_list(row['history'])
    candidates = parse_movie_list(row['candidates'])
    ground_truth = parse_movie_list(row['ground_truth'])

    # Prepare descriptive strings for user history
    user_history_desc = " ".join(history)

    # Combine ground truth and candidates
    all_movies = candidates

    # Shuffle for the prompt candidates
    prompt_candidates = all_movies[:]
    random.shuffle(prompt_candidates)

    ground_truth_movies = [movie for movie in candidates if movie in ground_truth]
    non_ground_truth_movies = [movie for movie in candidates if movie not in ground_truth]

    random.shuffle(ground_truth_movies)
    random.shuffle(non_ground_truth_movies)
    chosen_movies = ground_truth_movies + non_ground_truth_movies
    # Shuffle for the rejected list 
    rejected_movies = all_movies[:]
    random.shuffle(rejected_movies)

    # Generate meaningful reasons for `chosen`
    chosen_reasons = await generate_chosen_reasons(user_history_desc, chosen_movies)

    rejected_reasons = await generate_chosen_reasons(user_history_desc, rejected_movies)

    len_candidates = len(prompt_candidates)
    prompt = generate_prompt(
        user_history_desc,
        ", ".join(prompt_candidates),
        len_candidates
    )

    return {"prompt": prompt, 
            "chosen": "\n".join(chosen_reasons), 
            "rejected": "\n".join(rejected_reasons)}



# Process Dataset Incrementally
async def process_dataset_incrementally(input_path, output_path, dataset_name, batch_size=1000):
    dataset = pd.read_csv(input_path)
    processed_rows = []
    existed_length=0
    if os.path.exists(output_path):
        # 读取已存在的数据
        existed_dataset = pd.read_csv(output_path)
        existed_length = len(existed_dataset)
    
    for i in range(existed_length, len(dataset), batch_size):
        batch = dataset.iloc[i : i + batch_size]
        tasks = [process_row(row) for _, row in batch.iterrows()]
        results = await asyncio.gather(*tasks)

        # Append results to processed_rows
        processed_rows.extend(results)

        # append to existed dataset
        if existed_length == 0:
            pd.DataFrame(processed_rows).to_csv(output_path, index=False)
        else:
            pd.DataFrame(processed_rows).to_csv(output_path, index=False, mode='a', header=False)
        print(f"Processed {len(processed_rows)} rows and saved to {output_path}")


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
    output_csv = 'dataset/movielens_1M/dpo_dataset_2.csv'
    dataset_name = "sssssssshhhhhu/movielens_dpo_dataset_2"
 
    # Call the processing function
    await process_dataset_incrementally(input_csv, output_csv, dataset_name, batch_size=1000)

    # Call the Hugging Face upload function
    upload_to_huggingface(output_csv, dataset_name)

# Run the async function using asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())
    #python dpo_dataset_generate.py
    #huggingface-cli login
    