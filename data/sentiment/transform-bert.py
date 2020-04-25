import pandas as pd

if __name__ == "__main__":
    # TODO Stratified split
    df_train = pd.read_csv(f"original/train.csv").dropna().reset_index(drop=True)
    p = .8
    n = int(df_train.shape[0]*.8)
    df_train["text"] = df_train.apply(lambda row: row["sentiment"].strip()+" "+row["text"].strip(),1)
    df_train.loc[:n,["textID","text","selected_text"]].to_csv(f"original/train_data.txt",sep="\t",header=False, index=False)
    df_train.loc[n:,["textID","text","selected_text"]].to_csv(f"original/dev_data.txt",sep="\t",header=False, index=False)


    df_test = pd.read_csv(f"original/test.csv")
    df_test["text"] = df_test.apply(lambda row: row["sentiment"].strip()+" "+row["text"].replace("\t"," ").strip(),1)
    df_test["selected_text"] = ""
    df_test[["textID","text","selected_text"]].to_csv(f"original/test_data.txt",sep="\t",header=False, index=False)