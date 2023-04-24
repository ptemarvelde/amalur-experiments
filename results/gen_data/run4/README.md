Results with data generated with params (from `generate.py`):

```python
    joins = ['inner', 'left', 'outer']
    rows_T = [1000, 10000, 100000]
    columns_T = [10, 100]
    rho_r_R_list = [0.1, 0.5, 1.0]
    rho_c_S_list = [0.1, 0.5, 1.0]
    rho_c_R_list = [0.1, 0.5, 1.0]
    p_list = [0.0, 0.5]
    params = (rows_T, columns_T, rho_c_S_list, rho_c_R_list, rho_r_R_list, p_list)

    # left and outer join
    rho_r_S_list = [0.9]
```

Notes:
 - Run on a e2-highcpu-32 GCP VM. 32 cores, clockspeed: 2249.998 MHz
 - A dataset could not be generated for some combinations of parameters.
 - 
