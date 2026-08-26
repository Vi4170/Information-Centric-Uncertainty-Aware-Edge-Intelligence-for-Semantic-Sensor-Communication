\# CNN Embedding Novelty Detection



\## Objective



This module estimates how unusual a CWRU vibration observation is relative to

the reference distribution learned from the CNN training embeddings.



The module is intentionally separate from the VoI Engine.



\## Existing CNN Interface



The trained CNN provides a 64-dimensional learned embedding through:



```python

from src.cnn.model import extract\_embeddings

