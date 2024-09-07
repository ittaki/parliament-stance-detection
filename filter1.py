import pandas as pd
import re
import os

speeches_dir = r'..\ParlaMint-RS.txt\2020'
output_dir = r'output_2020'

os.makedirs(output_dir, exist_ok=True)

def read_meta_file(meta_file_path):
    with open(meta_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    header = lines[0].strip().split('\t')
    data = [line.strip().split('\t') for line in lines[1:]]
    return pd.DataFrame(data, columns=header)


def check_keywords(text):
    pattern = r'\b(EU|SSP|Evropska unija|Evropsk\w* unij\w*|Evrop\w*|Sporazum\* o stabilizaciji i pridruživanju)\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

for file_name in os.listdir(speeches_dir):
    if file_name.endswith('.txt') and not file_name.endswith('-meta.txt'):

        base_name = file_name[:-4]
        speeches_file_path = os.path.join(speeches_dir, file_name)
        meta_file_path = os.path.join(speeches_dir, base_name + '-meta.tsv')
        output_file_path = os.path.join(output_dir, f'{base_name}.csv')
        output_file_path_keywords = os.path.join(output_dir, f'{base_name}-keywords.csv')

        meta_df = read_meta_file(meta_file_path)

        meta_df['Speaker_birth'] = pd.to_numeric(meta_df['Speaker_birth'], errors='coerce')

        speeches_list = []
        with open(speeches_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    speech_id, text = parts
                    speeches_list.append({'ID': speech_id, 'Speech': text})

        speeches_df = pd.DataFrame(speeches_list)

        full_df = speeches_df.merge(meta_df, on='ID')
        full_df = full_df[full_df['Speech'].apply(lambda x: len(x) >= 450)]
        final_df = full_df[['Speaker_ID', 'Speaker_name', 'Speaker_party_name', 'Speaker_gender', 'Speaker_birth', 'Speech']]
        final_df = final_df.sort_values(by='Speaker_ID')
        final_df['mentions_EU'] = final_df['Speech'].apply(check_keywords)

        final_df = final_df[['Speaker_ID', 'Speaker_name', 'Speaker_party_name', 'Speaker_gender', 'Speaker_birth', 'mentions_EU', 'Speech']]

        eu_mentions_df = final_df[final_df['mentions_EU']]

        if not eu_mentions_df.empty:
            eu_mentions_df.to_csv(output_file_path_keywords, index=False, encoding='utf-8-sig')
            print(f"Filtered DataFrame with EU mentions saved for {file_name}. Preview:")
            print(eu_mentions_df.head())
        else:
            print(f"No EU mentions found in {file_name}, no CSV file created.")
