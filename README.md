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