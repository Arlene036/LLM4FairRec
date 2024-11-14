#!/bin/bash


models=("davinci-002") #  "gpt-3.5-turbo" "davinci-002"
mode=("easy" ) # "medium" "hard"

for model in "${models[@]}"; do
    for mode in "${mode[@]}"; do

        # python evaluation/cycle.py --model "$model" --mode "$mode" --prompt CoT 
        
        # python evaluation/connectivity.py --model "$model" --mode "$mode" --prompt none
        # python evaluation/connectivity.py --model "$model" --mode "$mode" --prompt k-shot
        # python evaluation/connectivity.py --model "$model" --mode "$mode" --prompt CoT
        # python evaluation/connectivity.py --model "$model" --mode "$mode" --prompt CoT  --SC 1 --SC_num 5

        python evaluation/flow.py --model "$model" --mode "$mode" --prompt none --token 1024
        python evaluation/flow.py --model "$model" --mode "$mode" --prompt k-shot --token 1024
        python evaluation/flow.py --model "$model" --mode "$mode" --prompt CoT --token 1024
        python evaluation/flow.py --model "$model" --mode "$mode" --prompt CoT --SC 1 --SC_num 5 --token 1024

        # python evaluation/cycle.py --model "$model" --mode "$mode" --prompt k-shot
        # python evaluation/cycle.py --model "$model" --mode "$mode" --prompt CoT
        # python evaluation/cycle.py --model "$model" --mode "$mode" --prompt CoT  --SC 1 --SC_num 5
        
      
    done
done