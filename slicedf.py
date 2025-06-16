import pandas as pd

def slice_dataset(input_file, output_file, sample_size=0.1, random_seed=42):
    df = pd.read_csv(input_file, encoding='ISO-8859-1', header=None)
    sampled_df = df.sample(frac=sample_size, random_state=random_seed)    
    sampled_df.to_csv(output_file, index=False, header=False)
    return len(sampled_df)

input_file = "training.1600000.processed.noemoticon.csv"
output_file = "sliced_data.csv"
    
rows_sampled = slice_dataset(input_file, output_file)
print(f"Successfully sampled {rows_sampled} rows to {output_file}")