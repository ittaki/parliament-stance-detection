import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
import joblib
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from huggingface_hub import login
import matplotlib.pyplot as plt
from collections import defaultdict

ideological_groups = {
    "Skrajno levi": ["Pokret socijalista", "Komunistička partija"],
    "Srednje levi": [
        "Socijalistička partija Srbije", 
        "Stranka slobode i pravde", 
        "Socijaldemokratska stranka", 
        "Socijaldemokratska partija Srbije", 
        "Partija ujedinjenih penzionera Srbije", 
        "Nova stranka", 
        "Liga socijaldemokrata Vojvodine", 
        "Zelena stranka", 
        "Zeleni Srbije", 
        "Demokratska stranka"
    ],
    "Srednje desni": [
        "Srpska napredna stranka", 
        "Bolja Srbija", 
        "Zajedno za Srbiju", 
        "Ujedinjena seljačka stranka", 
        "Srpski pokret obnove", 
        "Srpska narodna partija", 
        "Pokret Snaga Srbije – BK", 
        "Nova Srbija", 
        "Narodna seljačka stranka", 
        "Demokratska stranka Srbije"
    ],
    "Skrajno desni": [
        "Srpska radikalna stranka", 
        "Jedinstvena Srbija", 
        "Srpski pokret Dveri", 
        "Dosta je bilo"
    ],
    "Ostalo": [
        "Savez vojvođanskih Mađara", 
        "Partija za demokratsko delovanje", 
        "Demokratski savez Hrvata u Vojvodini", 
        "Liberalno-demokratska partija", 
        "Stranka demokratske akcije Sandžaka", 
        "Stranka pravde i pomirenja", 
        "Brez"
    ]
}


# OVERSAMPLING
def oversample_classes(train_data, label_encoder, target_class):
    target_class_index = label_encoder.transform([target_class])[0]
    class_counts = train_data['labels'].value_counts()

    max_count = max(class_counts)
    
    sampling_strategy = {
        label_index: max(class_counts[label_index], class_counts[target_class_index])
        for label_index in class_counts.index
    }

    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    train_data_resampled, train_labels_resampled = ros.fit_resample(train_data, train_data['labels'])
    train_data_resampled['labels'] = train_labels_resampled
    return train_data_resampled



def check_token_lengths(dataset, tokenizer, max_length=512):
    exceeding_entries = []
    for index, row in dataset.iterrows():
        prompt = row['Relevant_Speech'] + "[SEP]Stav prema Evropi, pridruživanju Evropskoj uniji i generalno evropskim vrednostima je [MASK]."
        
        tokenized = tokenizer.encode(prompt, truncation=False)
        if len(tokenized) > max_length:
            exceeding_entries.append((index, len(tokenized), prompt))
    return exceeding_entries


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def print_device_info():
    print(f"Using {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


print_device_info()


hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)

combined_data = pd.read_csv('EU_final_skrajsan.csv')


if 'Speech' in combined_data.columns:
    combined_data = combined_data.drop(columns=['Speech'])

def categorize_age(birth_year):
    if birth_year >= 1940 and birth_year <= 1960:
        return '1940-1960'
    elif birth_year >= 1961 and birth_year <= 1970:
        return '1961-1970'
    elif birth_year >= 1971 and birth_year <= 1980:
        return '1971-1980'
    elif birth_year >= 1981:
        return '1981+'
    else:
        return None

combined_data['Age_Group'] = combined_data['Speaker_birth'].apply(lambda x: categorize_age(int(x)) if pd.notna(x) and str(x).isdigit() else None)
combined_data = combined_data.dropna(subset=['Age_Group'])

# Encode labels
label_encoder = LabelEncoder()
combined_data['Stance'] = combined_data['Stance'].replace({'NEUTRALNO': 'NEUTRALNO', 'ZA': 'ZA', 'PROTIV': 'PROTIV'})
combined_data['labels'] = label_encoder.fit_transform(combined_data['Stance'])
joblib.dump(label_encoder, 'label_encoder.joblib')

# Replace "-" with "Ostalo" in the Speaker_party_name column
combined_data['Speaker_party_name'] = combined_data['Speaker_party_name'].replace('-', 'Ostalo')


undefined_gender = combined_data[combined_data['Speaker_gender'] == 'U']
if not undefined_gender.empty:
    print("Rows with undefined gender (U):")
    print(undefined_gender)
combined_data = combined_data[combined_data['Speaker_gender'].isin(['M', 'F'])]


model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

# Check token lengths
exceeding_entries = check_token_lengths(combined_data, tokenizer)

if exceeding_entries:
    print(f"Found {len(exceeding_entries)} entries exceeding {tokenizer.model_max_length} tokens:")
    for index, length, prompt in exceeding_entries:
        print(f"Index {index} exceeds with {length} tokens. Prompt: {prompt[:100]}...")
else:
    print("No entries exceed the tokenization limit.")


# StratifiedKFold
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_predictions = []
all_true_labels = []
all_relevant_speeches = [] 
all_indices = [] 
metrics_per_fold = []

# K-fold Cross Validation
for fold, (train_idx, test_idx) in enumerate(skf.split(combined_data, combined_data['labels'])):
    print(f"Fold {fold + 1}/{n_splits}")
    
    
    train_data = combined_data.iloc[train_idx]
    test_data = combined_data.iloc[test_idx]

    train_data = oversample_classes(train_data, label_encoder, target_class='NEUTRALNO')

    train_speeches = set(train_data['Relevant_Speech'])
    test_speeches = set(test_data['Relevant_Speech'])
    overlaps = train_speeches.intersection(test_speeches)

    if overlaps:
        print(f"Overlap found in fold {fold + 1}: {len(overlaps)} overlapping speeches.")
        print(overlaps)
    else: 
        print("No overlaps found")

    # convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

    train_dataset = train_dataset.remove_columns('__index_level_0__') if '__index_level_0__' in train_dataset.column_names else train_dataset
    test_dataset = test_dataset.remove_columns('__index_level_0__') if '__index_level_0__' in test_dataset.column_names else test_dataset

    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

    # construct prompts
    def construct_prompt(examples):
        prompts = []
        for i in range(len(examples['Relevant_Speech'])):
            prompt = examples['Relevant_Speech'][i] + "[SEP]Stav prema Evropi, pridruživanju Evropskoj uniji i generalno evropskim vrednostima je [MASK]."
            prompts.append(prompt)
        examples['prompt'] = prompts
        return examples
    
    datasets = datasets.map(construct_prompt, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token, num_labels=len(label_encoder.classes_), device_map="balanced")

    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # tokenizing dataset with prompts
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples['prompt'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = examples["labels"]
        return tokenized_inputs

    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["prompt"])

    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_datasets = tokenized_datasets.remove_columns(["Speaker_birth", "Speaker_ID", "Speaker_name", "Speaker_party_name", "Speaker_gender", "Stance", "Age_Group"])

    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold + 1}',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss', 
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    # evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation Results for fold {fold + 1}:", eval_results)

    # predict on the test set
    model.eval() 
    predictions = trainer.predict(tokenized_datasets['test'])
    predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

    # all predictions, true labels, and relevant speeches
    all_predictions.extend(predicted_labels)
    all_true_labels.extend(test_data['labels'].values)
    all_relevant_speeches.extend(test_data['Relevant_Speech'].values)  # Collect relevant speeches
    all_indices.extend(test_idx)

    # decode labels and generate classification report
    true_labels = test_data['labels'].values
    predicted_stances = label_encoder.inverse_transform(predicted_labels)
    true_stances = label_encoder.inverse_transform(true_labels)

    target_names = list(label_encoder.classes_)

    # calculate metrics for this fold
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    accuracy = np.mean(predicted_labels == true_labels)
    metrics_per_fold.append({'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy})

    # classification report
    report = classification_report(true_stances, predicted_stances, target_names=target_names)
    print(f"Classification report for fold {fold + 1}:\n", report)

# overall precision, recall, f1-score, and accuracy across all folds
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))

print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")
print(f"Overall F1-Score: {f1}")
print(f"Overall Accuracy: {accuracy}")

# variance for each metric across folds
metrics_df = pd.DataFrame(metrics_per_fold)
variance = metrics_df.var()
print(f"Variance of metrics across folds:\n{variance}")

############################################################################################
# CONFUSION MATRIX:
true_stances = label_encoder.inverse_transform(all_true_labels)
predicted_stances = label_encoder.inverse_transform(all_predictions)

true_stances = [label.replace('NEUTRALNO', 'NEVTRALNO').replace('PROTIV', 'PROTI') for label in true_stances]
predicted_stances = [label.replace('NEUTRALNO', 'NEVTRALNO').replace('PROTIV', 'PROTI') for label in predicted_stances]


cm = confusion_matrix(true_stances, predicted_stances, labels=['NEVTRALNO', 'PROTI', 'ZA'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NEVTRALNO', 'PROTI', 'ZA'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)

plt.xlabel('Napovedani razred')
plt.ylabel('Dejanski razred')

plt.savefig('/shared/home/andela.rajovic/eu/xml_roberta/confusion_matrix_eu_slovenian.png')
plt.show()

############################################################################################

# Combine predictions, true labels, and relevant speeches into a DataFrame
results_df = pd.DataFrame({
    'Relevant_Speech': all_relevant_speeches,
    'True_Stance': label_encoder.inverse_transform(all_true_labels),
    'Predicted_Stance': label_encoder.inverse_transform(all_predictions)
})


results_df.to_csv('/shared/home/andela.rajovic/eu/xml_roberta/model_predictions_analysis.csv', index=False)



combined_data.loc[all_indices, 'predicted_stances'] = label_encoder.inverse_transform(all_predictions)


combined_data['predicted_stances'] = combined_data['predicted_stances'].replace({'PROTIV': 'PROTI'})
combined_data['predicted_stances'] = combined_data['predicted_stances'].replace({'NEUTRALNO': 'NEVTRALNO'})
combined_data['Speaker_gender'] = combined_data['Speaker_gender'].replace({'M': 'Moški', 'F': 'Ženske'})

if combined_data['Speaker_gender'].isna().any():
    print("NaN values detected in 'Speaker_gender' after replacement.")


ostalo_entries = combined_data[combined_data['Speaker_party_name'] == 'Brez']
print("Vrstice z oznako 'Brez' v stolpcu 'Speaker_party_name':")
print(ostalo_entries)


def map_party_to_group(party_name):
    for group, parties in ideological_groups.items():
        if party_name in parties:
            return group
    return "Unknown"

combined_data['Ideological_Group'] = combined_data['Speaker_party_name'].apply(map_party_to_group)

mapped_parties = combined_data[['Speaker_party_name', 'Ideological_Group']].drop_duplicates()
print("Mapping of parties to ideological groups:")
print(mapped_parties.sort_values(by='Ideological_Group'))

unknown_parties = combined_data[combined_data['Ideological_Group'] == 'Unknown']['Speaker_party_name'].unique()
print("Unknown parties:", unknown_parties)



# PARTY DISTRIBUTION
def plot_stance_distribution_party(data, output_path):
    if data.empty:
        print("No data available for plotting")
        return

    grouped_ideology = data.groupby(['Ideological_Group', 'predicted_stances']).size().unstack(fill_value=0)
           
    ordered_groups = ['Skrajno levi', 'Srednje levi', 'Srednje desni', 'Skrajno desni', 'Ostalo']
    grouped_ideology = grouped_ideology.reindex(ordered_groups, fill_value=0)

    grouped_ideology_percentage = grouped_ideology.div(grouped_ideology.sum(axis=1), axis=0) * 100

    grouped_ideology_percentage = grouped_ideology_percentage.reindex(ordered_groups)

    stance_colors = {'ZA': 'lightgreen', 'PROTI': 'red', 'NEVTRALNO': 'lightblue'}

   
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25  
    x = np.arange(len(grouped_ideology_percentage.index))  

    for i, stance in enumerate(grouped_ideology_percentage.columns):
        ax.bar(x + i * width, grouped_ideology_percentage[stance], width, label=stance, color=stance_colors[stance])

    ax.set_xlabel('Ideološka skupina')
    ax.set_ylabel('Odstotek')
    ax.set_title('Porazdelitev stališč po ideoloških skupinah')
    ax.set_xticks(x + width)
    ax.set_xticklabels(grouped_ideology_percentage.index)
    ax.legend(title='Stališče')
    ax.set_ylim(0, 100) 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

plot_stance_distribution_party(combined_data, '/shared/home/andela.rajovic/eu/xml_roberta/stance_distribution_by_party_grouped.png')



# GENDER GROUP
def plot_gender_distribution_separate_bars(data, output_path):
    if data.empty:
        print("No data available for plotting")
        return
    
    grouped_gender = data.groupby(['Speaker_gender', 'predicted_stances']).size().unstack(fill_value=0)

    grouped_gender_percentage = grouped_gender.div(grouped_gender.sum(axis=1), axis=0) * 100

    stance_colors = {'ZA': 'lightgreen', 'PROTI': 'red', 'NEVTRALNO': 'lightblue'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.18  
    x = np.arange(len(grouped_gender_percentage.index))  

    for i, stance in enumerate(grouped_gender_percentage.columns):
        ax.bar(x + i * width, grouped_gender_percentage[stance], width, label=stance, color=stance_colors[stance])

    ax.set_xlabel('Spol')
    ax.set_ylabel('Odstotek')
    ax.set_title('Porazdelitev stališč po spolu')
    ax.set_xticks(x + width)
    ax.set_xticklabels(grouped_gender_percentage.index)
    ax.legend(title='Stališče')
    ax.set_ylim(0, 100)  
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

plot_gender_distribution_separate_bars(combined_data, '/shared/home/andela.rajovic/eu/xml_roberta/stance_distribution_by_gender_separate_bars.png')



# AGE GROUPS
def plot_age_group_distribution_separate_bars(data, output_path):
    if data.empty:
        print("No data available for plotting")
        return

    grouped_age = data.groupby(['Age_Group', 'predicted_stances']).size().unstack(fill_value=0)

    grouped_age_percentage = grouped_age.div(grouped_age.sum(axis=1), axis=0) * 100

    stance_colors = {'ZA': 'lightgreen', 'PROTI': 'red', 'NEVTRALNO': 'lightblue'}

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25  
    x = np.arange(len(grouped_age_percentage.index)) 

    for i, stance in enumerate(grouped_age_percentage.columns):
        ax.bar(x + i * width, grouped_age_percentage[stance], width, label=stance, color=stance_colors[stance])

    ax.set_xlabel('Starostna skupina')
    ax.set_ylabel('Odstotek')
    ax.set_title('Porazdelitev stališč po starostnih skupinah')
    ax.set_xticks(x + width)
    ax.set_xticklabels(grouped_age_percentage.index)
    ax.legend(title='Stališče')
    ax.set_ylim(0, 100)  
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

plot_age_group_distribution_separate_bars(combined_data, '/shared/home/andela.rajovic/eu/xml_roberta/stance_distribution_by_age_group_separate_bars.png')



