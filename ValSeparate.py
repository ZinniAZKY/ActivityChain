import pandas as pd
from sklearn.model_selection import train_test_split

with open("/home/ubuntu/Documents/TokyoPT/PTChain/Chukyo2011PTChainRefined.txt", 'r') as file:
    lines = file.read().splitlines()

df = pd.DataFrame(lines, columns=['text'])
df = df.sample(frac=1, random_state=42)
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

train_df.to_csv("/home/ubuntu/Documents/TokyoPT/PTChain/train.txt",
                header=False,
                index=False)
val_df.to_csv("/home/ubuntu/Documents/TokyoPT/PTChain/eval.txt",
              header=False,
              index=False)


