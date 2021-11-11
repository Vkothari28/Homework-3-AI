import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.e**(-x))


df = pd.read_csv('../input/chronic-kidney-disease/new_model.csv')