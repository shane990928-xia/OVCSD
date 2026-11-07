# OVCSD Transformers runner

For the ALFWorld paper configuration, use the [verl training guide](../README.md). The `ovcsd/` package also provides a standalone Transformers runner for WebShop.

## Quick start

Run from the repository root. The offline tests use a randomly initialized language model on CPU and exercise generation, intervention, suffix distillation and an optimizer step.

```bash
git clone https://github.com/shane990928-xia/OVCSD.git
cd OVCSD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Training and evaluation

Set the model path to a task SFT checkpoint with its tokenizer and chat template.

### WebShop

Use the full human-goal catalog and Lucene index. Install the upstream benchmark in its own Python 3.8 environment with Java. Before running the upstream `setup.sh -d all`, set the two active defaults in `web_agent_site/utils.py` to `items_ins_v2.json` and `items_shuffle.json` as described in upstream setup step 6. The upstream index builder reads those paths during setup. [WebShop setup](https://github.com/princeton-nlp/WebShop/tree/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd)

```bash
export WEBSHOP=/path/to/full/WebShop
export SHOP_PY=/path/to/webshop-env/bin/python
export SHOP_SFT=/path/to/webshop-sft
python -m ovcsd.run train webshop --model "$SHOP_SFT" \
  --env-python "$SHOP_PY" --resource "$WEBSHOP" --output runs/shop-train
python -m ovcsd.run eval webshop --model runs/shop-train/step-100 \
  --env-python "$SHOP_PY" --resource "$WEBSHOP" --output runs/shop-test
```

Training uses integer goal IDs from 1500 onward. Test uses IDs 0–499. Graded terminal rewards contribute to the evaluation score × 100. Exact terminal success supplies the binary training reward in both benchmarks.
