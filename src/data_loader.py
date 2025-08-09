from sklearn.datasets import load_iris
import pandas as pd
import os

def load_and_save_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)

if __name__ == "__main__":
    load_and_save_data()
