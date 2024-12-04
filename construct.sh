# construct the natural language dataset from a 3 column dataset for inference
# input: 3 column dataset for inference (STRICTLY COLUMN NAMES: user_id, groundtruth, top10_recommendations)
# output: natural language dataset for further inference

python construct_nl_data.py \
    --input_path /Users/lavander/cpan/h411711/cf/item_cf_recommendations.csv \
    --output_path /Users/lavander/cpan/h411711/cf/cf_natural_language_dataset.csv \
    --n 1000

python construct_nl_data.py \
    --input_path /Users/lavander/cpan/h411711/RippleNet-PyTorch-baseline/src/results/top_k_recommendations.csv \
    --output_path /Users/lavander/cpan/h411711/RippleNet-PyTorch-baseline/src/results/ripple_natural_language_dataset.csv \
    --n 1000    