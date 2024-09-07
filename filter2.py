import pandas as pd
import os

output_filtered_dir = r'output_filtered_2015'
combined_output_path = r'combined_filtered_speeches_2015.csv'

combined_df = pd.DataFrame()

for file_name in os.listdir(output_filtered_dir):
    if file_name.endswith('-keywords-samoMIGRACIJE.csv'):
        file_path = os.path.join(output_filtered_dir, file_name)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.to_csv(combined_output_path, index=False, encoding='utf-8-sig')

print(f"Combined DataFrame saved to {combined_output_path}. Preview:")
print(combined_df.head())
