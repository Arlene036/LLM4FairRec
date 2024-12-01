from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import argparse
from tqdm import tqdm
import pandas as pd
import asyncio
from utils import dispatch_openai_requests

def format_prompt(history, candidates):
    len_candidates = len(candidates.split(','))
    base_prompt = """You are a recommender system. Based on a user's historical likes and dislikes, rank the given candidate movies by their likelihood of being the user's next favorite, according to their watching history. Please think step by step.

This user's historical interactions: {history}
There are {len_candidates} Candidates for recommendation: {candidates}

Strictly follow the output format:
Rank1: movieid - Reason: explain why the user would most likely enjoy this movie
Rank2: movieid - Reason: explain why the user would likely enjoy this movie second
Rank3: ...
...
Rank{len_candidates}: movieid - Reason: explain why the user would like this movie last
Please provide a ranked list of the recommended movies. You MUST rank only the given candidates and cannot include any movies not listed in the candidate list."""
    
    return base_prompt.format(len_candidates=len_candidates, history=history, candidates=candidates)

def generate_recommendations(model, tokenizer, prompt, max_length=2048, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main(args):
    data = pd.read_csv(args.input_path)
    results = []

    if "gpt" not in args.model_name:
        print("inference model start")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        for index, item in tqdm(data.iterrows()):
            history = item['history']
            candidates = item['candidates']
            ground_truth = item.get('ground_truth', [])
            
            prompt = format_prompt(history, candidates)
            response = generate_recommendations(model, tokenizer, prompt, args.max_tokens, args.temperature)
            
            results.append({
                'history': history,
                'candidates': candidates,
                'ground_truth': ground_truth,
                'recommendation': response
            })
            with open(args.output_path, 'a') as f:
                json.dump(results, f, indent=2)
    else:
        print("GPT model start")
        history_list = data['history']
        candidates_list = data['candidates']
        ground_truth_list = data['ground_truth']
        prompt_list = [format_prompt(str(history), str(candidates)) for history, candidates in zip(history_list, candidates_list)]
        response_list = asyncio.run(
            dispatch_openai_requests(
                messages_list=prompt_list,
                model=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        )

        with open(args.output_path, 'a') as f:
            for history, candidates, ground_truth, response in zip(history_list, candidates_list, ground_truth_list, response_list):
                results.append({
                    'history': history, 'candidates': candidates, 'ground_truth': ground_truth, 'recommendation': response
                })
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='model name') # mistralai/Mistral-7B-v0.1
    parser.add_argument('--temperature', type=float, default=0.3, help='temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='max tokens')
    parser.add_argument('--nl_input_path', type=str, default='../dataset/movielens_1M/natural_language_top10_sample.csv', 
                        help='Natural language dataset')
    parser.add_argument('--tabu_input_path', type=str, default='../dataset/movielens_1M/sequential_top10_filtered.csv.csv',
                        help='sequential dataset') # TODO, 修改output的内容更简单
    parser.add_argument('--output_path', type=str, default='inference_results/results.txt', help='inference results path')
    args = parser.parse_args()
    
    main(args)