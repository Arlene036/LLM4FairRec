import pandas as pd
import asyncio
import random
from openai import AsyncOpenAI
from datasets import Dataset, DatasetDict
import re
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
    Generate reasons for chosen reranking of movie candidates used for DPO training.

    User's movie watching history: {user_history}
 
    -Reranked movies:
    {chosen_request}

    Explain why the following movies are ranked in their respective positions and match the user's preferences,one movie in a line.
    Possible responses for chosen reranking reason could be:
    This movie combines science fiction, thriller, and mystery elements, aligning strongly with the user's preference for complex, thought-provoking stories as seen in their history.
    The mathematical genius theme and psychological intensity resonate with the user's interest in thrilling and intelligent narratives.
    The movie tells a romantic story happening in high school which is similar to user's interested movie 56.

    Please STRICTLY follow the format below for output, don't output any other words, do not repeatedly output one movie, just one movie a line with the reason, do not output all the movie information except the id
    Rank 1: Movie ..  - Reason: ...
    Rank 2: Movie ..  - Reason: ...
    ...
    """
    #print(prompt)
    # Call OpenAI and handle response
    response = await dispatch_openai_requests(
        [prompt], model=model, temperature=0.7, max_tokens=500
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
    user_history_desc = ", ".join(history)

    # Combine ground truth and candidates
    all_movies = candidates

    # Shuffle for the prompt candidates
    prompt_candidates = all_movies[:]
    random.shuffle(prompt_candidates)

    chosen_movies = [movie for movie in candidates if movie in ground_truth] + \
                    [movie for movie in candidates if movie not in ground_truth]

    # Shuffle for the rejected list
    rejected_movies = all_movies[:]
    random.shuffle(rejected_movies)

    # Generate meaningful reasons for `chosen`
    chosen_reasons = await generate_chosen_reasons(user_history_desc, all_movies)
    chosen_reasons_map = {}
    for reason in chosen_reasons:
        match = re.match(r"Rank \d+: Movie (\d+) - Reason: (.+)", reason)
        if match:
            movie_id = match.group(1)
            movie_reason = match.group(2)
            chosen_reasons_map[movie_id] = movie_reason

    # Format rejected reasons based on rejected_movies order
    rejected_reasons = []
    for i, movie in enumerate(rejected_movies):
        match = re.match(r"Movie (\d+):", movie)
        if match:
            movie_id = match.group(1)
            reason = chosen_reasons_map.get(movie_id, "No reason found")
            rejected_reasons.append(f"Rank {i + 1}: Movie {movie_id} - Reason: {reason}")

    # # Format `chosen` output
    # chosen = [
    #     f"chosen_reasons[i]"
    #     for i in range(len(all_movies))
    #     if i < len(chosen_reasons)
    # ]

    # # Format `rejected` output
    # rejected = [
    #     f"Rank{i + 1}: {rejected_movies[i]} - Reason: {rejected_reasons[i]}"
    #     for i in range(len(rejected_movies))
    #     if i < len(rejected_reasons)
    # ]

    # Generate prompt with shuffled candidates
    len_candidates = len(prompt_candidates)
    prompt = generate_prompt(
        user_history_desc,
        ", ".join(prompt_candidates),
        len_candidates
    )

    return {"prompt": prompt, "chosen": "\n".join(chosen_reasons), "rejected": "\n".join(rejected_reasons)}




# Process Dataset Incrementally
async def process_dataset_incrementally(input_path, output_path, batch_size=1000):
    dataset = pd.read_csv(input_path)
    processed_rows = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i : i + batch_size]
        tasks = [process_row(row) for _, row in batch.iterrows()]
        results = await asyncio.gather(*tasks)

        # Append results to processed_rows
        processed_rows.extend(results)

        # Save incrementally to avoid data loss
        pd.DataFrame(processed_rows).to_csv(output_path, index=False)
        print(f"Processed {len(processed_rows)} rows and saved to {output_path}")


# Upload to Hugging Face
def upload_to_huggingface(output_path, dataset_name):
    processed_df = pd.read_csv(output_path)
    hf_dataset = Dataset.from_pandas(processed_df)
    DatasetDict({"train": hf_dataset}).push_to_hub(dataset_name)

import asyncio

# Define your main async function
async def main():
    # Call your async function here
    input_csv = 'dataset/movielens_1M/natural_language_top10_sample.csv'
    output_csv = 'dataset/movielens_1M/dpo_dataset.csv'
    dataset_name = "sssssssshhhhhu/movielens_dpo_dataset"
 
    # Call the processing function
    await process_dataset_incrementally(input_csv, output_csv, batch_size=100)

    # Call the Hugging Face upload function
    upload_to_huggingface(output_csv, dataset_name)

# Run the async function using asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())
    #python dpo_dataset_generate.py
    #huggingface-cli login
    