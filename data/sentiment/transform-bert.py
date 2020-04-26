import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # TODO Stratified split
    df_train = pd.read_csv(f"original/train.csv").dropna().reset_index(drop=True)
    df_train["text"] = df_train.apply(lambda row: row["sentiment"].strip()+" "+row["text"].strip(),1)
    p = .7
    df_train, df_dev = train_test_split(df_train, test_size = 1 - p, random_state=42, stratify=df_train["sentiment"])
    df_train.loc[:,["textID","text","selected_text"]].to_csv(f"original/train_data.txt",sep="\t",header=False, index=False)
    df_dev.loc[:,["textID","text","selected_text"]].to_csv(f"original/dev_data.txt",sep="\t",header=False, index=False)


    df_test = pd.read_csv(f"original/test.csv")
    df_test["text"] = df_test.apply(lambda row: row["sentiment"].strip()+" "+row["text"].replace("\t"," ").strip(),1)
    df_test["selected_text"] = ""
    df_test[["textID","text","selected_text"]].to_csv(f"original/test_data.txt",sep="\t",header=False, index=False)