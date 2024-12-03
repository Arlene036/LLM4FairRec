# construct the natural language dataset from a 3 column dataset for inference
# input: 3 column dataset for inference (STRICTLY COLUMN NAMES: user_id, groundtruth, top10_recommendations)
# output: natural language dataset for further inference

python construct_nl_data.py \
    --input_path YOUR_INPUT_PATH.csv \
    --output_path natural_language_dataset.csv \
    --n 1000