import json 
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import prepareUniqueTestData



def loadGPTCacheDataset():
    with open("similiar_qqp_full.json", "r") as mock_file:
        file_content = mock_file.read()
        mock_data = json.loads(file_content)
        df = pd.DataFrame(mock_data)
        df.columns = ['question1', 'question2', 'is_duplicate']
        return df



def  trainValTestSplits(df):
    # Assuming `df` is your DataFrame and you want a 70-15-15 split for train-test-validation
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    test, val = train_test_split(temp, test_size=0.5, random_state=42)
    # len of train, test, val
    print(len(train), len(test), len(val))

    return train, val, test



def main():
    df = loadGPTCacheDataset()
    # df = prepareUniqueTestData(df)

    label1_df = df[df['is_duplicate'] == 1]
    label0_df = df[df['is_duplicate'] == 0]

    
    label1_train, label1_val, label1_test = trainValTestSplits(label1_df)
    label0_train, label0_val, label0_test = trainValTestSplits(label0_df)

    train = pd.concat([label1_train, label0_train])
    val = pd.concat([label1_val, label0_val])
    test = pd.concat([label1_test, label0_test])

    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    test_label1_df = test[test['is_duplicate'] == 1]
    test_label0_df = test[test['is_duplicate'] == 0]

    diff = len(test_label0_df) - len(test_label1_df)

    test = pd.concat([test_label1_df, test_label0_df.head(len(test_label1_df))])


    # assert that all data is used
    assert len(train) + len(val) + len(test) + diff == len(df)



    train.to_csv("../dataset_gptcache/train.csv", index=False)
    val.to_csv("../dataset_gptcache/val.csv", index=False)
    test.to_csv("../dataset_gptcache/test.csv", index=False)



main()
