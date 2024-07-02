import os
os.environ['CUDA_VISIBLE_DEVICE'] = '1'

import numpy as np
import transformers
from omnixai.data.text import Text
from omnixai.explainers.nlp import NLPExplainer

x = Text([
    "What a great movie!",
    "The Interview was neither that funny nor that witty. "
    "Even if there are words like funny and witty, the overall structure is a negative type."
])

# The preprocessing function
preprocess = lambda x: x.values
# A transformer model for sentiment analysis
model = transformers.pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    return_all_scores=True
)
# The postprocessing function
postprocess = lambda outputs: np.array([[s["score"] for s in ss] for ss in outputs])


explainer = NLPExplainer(
    explainers=["polyjuice"],
    mode="classification",
    model=model,
    preprocess=preprocess,
    postprocess=postprocess
)
local_explanations = explainer.explain(x)
print("Counterfactual results:")
local_explanations['polyjuice'].ipython_plot()


