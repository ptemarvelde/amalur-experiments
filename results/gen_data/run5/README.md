Run with small number of generated datasets (192).

quite a lot of 'row out of bounds' while generating (only when rho_r_R = 0.1?)

params:
```python
    joins = ['inner', 'left', 'outer']
    rows_T = [1000, 10000]
    columns_T = [10, 100]
    rho_r_R_list = [0.1, 0.8]
    rho_c_S_list = [0.1, 0.8]
    rho_c_R_list = [0.1, 0.8]
    p_list = [0.0, 0.5]
    params = (rows_T, columns_T, rho_c_S_list, rho_c_R_list, rho_r_R_list, p_list)

    # left and outer join
    rho_r_S_list = [0.9]
```
