import pandas as pd

def clean_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.columns.str.strip()
    df.replace({'?': None}, inplace=True)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == '__main__':
    print("Cleaning data")
    df = clean_data("data/census.csv", "data/clean_data.csv")
    print("Done")
