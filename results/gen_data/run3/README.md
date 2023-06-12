Very incomplete run. quit after error like: 
```
Dataset progression:   1%|          | 8/972 [26:42<53:38:11, 200.30s/it]
Traceback (most recent call last):
  File "/user/src/app/amalur-factorization/experiment.py", line 392, in <module>
    run_experiments()
  File "/user/src/app/amalur-factorization/experiment.py", line 97, in run_experiments
    model_list = create_model_tuples(models, amalur_matrix)
  File "/user/src/app/amalur-factorization/experiment.py", line 38, in create_model_tuples
    model_list.append((NM.materialize(), 'materialized'))
  File "/user/src/app/amalur-factorization/amalur/AmalurMatrix.py", line 84, in materialize
    result += multi_dot([self.I[k], self.S[k], self.M[k].T])
  File "/opt/conda/lib/python3.10/site-packages/scipy/sparse/_base.py", line 468, in __add__
    raise ValueError("inconsistent shapes")
```