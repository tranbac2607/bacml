import pandas as pd 

def read_csv(filename):
    df = pd.read_csv(filename).values
    x = df[:,0].reshape(-1, 1)
    y = df[:,1].reshape(-1, 1)
    return x, y