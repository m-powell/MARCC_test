import numpy as np
import pandas as pd

# In the following, P(Xj=x|Y=y) is accessed with p_xj_y[y,x].
p_x1_y = np.array([[.9, .1], [.1, .9]])
p_x2_y = np.array([[.95, .05], [.2, .8]])
p_x3_y = np.array([[.99, .01], [.29, .71]])
p_x4_y = np.array([[.5, .5], [.5, .5]])
p_y = np.array([.5, .5])

pmf = np.ones((2**5,6)) * -1

pmf_idx = 0
for x1 in [0,1]:
    for x2 in [0,1]:
        for x3 in [0,1]:
            for x4 in [0,1]:
                for y in [0,1]:
                    p = p_x1_y[y, x1] * p_x2_y[y, x2] * p_x3_y[y, x3] * p_x4_y[y, x4] * p_y[y]
                    pmf[pmf_idx, :] = np.array([x1, x2, x3, x4, y, p])
                    pmf_idx += 1

pmf = pd.DataFrame(pmf)
pmf.columns = ["X1", "X2", "X3", "X4", "Y", "p"]

pmf.to_csv("marcc_test_pmf.csv", index = False)