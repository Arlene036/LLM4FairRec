# LLM4RecAgent

A research project on recommendation system intelligent agents based on Large Language Models.

## Paper Reproduction


We reproduce the paper ["Can Language Models Solve Graph Problems in Natural Language?"](https://arxiv.org/abs/2305.10037) (NeurIPS 2023 Spotlight).

The original implementation can be found at [Arthur-Heng/NLGraph](https://github.com/Arthur-Heng/NLGraph).

### Runing Instruction

```
git clone --recursive https://github.com/Arlene036/LLM4RecAgent.git
pip install -r requirements.txt
export OPENAI_API_KEY=...
cd baselines
bash eval.sh
```



## Inference from "3 Columns data"

1. Prepare your "3 columns dataset", which should be a csv file, STRICTLY COLUMN NAMES: user_id, groundtruth, top10_recommendations.

2. run
   ```
   bash construct.sh
   ```

3. Go to Colab, connect to T4 GPU
   https://colab.research.google.com/drive/1iXGHxfmJSNDD3Z39f08oRTJyhEvy2s3u?usp=sharing
   