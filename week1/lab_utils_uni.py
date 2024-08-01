import numpy as np
from lab_utils_common import compute_cost
from ipywidgets import interact

def plt_intuition(x_train, y_train):
  w_range = np.array(0, 400)
  w_array = np.arange(*w_range, 5)
  cost = np.zeros(len(w_array))
  tmp_b = 100

  for i in w_array:
    tmp_w = w_array[i]
    cost[i] = compute_cost(x_train, y_train, tmp_w, tmp_b)

  # @interact()