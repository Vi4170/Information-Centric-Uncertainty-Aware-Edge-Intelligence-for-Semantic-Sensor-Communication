\# CNN-Based Uncertainty Estimation — Baseline Report



\## Objective



This module estimates prediction uncertainty from the probability distribution

produced by the existing trained CWRU CNN.



The module is separate from the Novelty Detection module and the VoI Engine.



\## Existing CNN Interface



The trained CNN produces a four-class softmax probability vector for each

2048-sample vibration observation.



The existing prediction interface is reused:



```python

from src.cnn.model import predict\_probabilities

