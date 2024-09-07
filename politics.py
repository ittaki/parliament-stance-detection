import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
import joblib
from imblearn.over_sampling import RandomOverSampler
import matplotlib.patches as mpatches
import numpy as np

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
        prompt = row['Relevant_Speech'] + "[SEP]Stance towards Europian Union (EU), joining the Europian union and generally europian vales is [MASK]."
        tokenized = tokenizer.encode(prompt, truncation=False)
        if len(tokenized) > max_length:
            exceeding_entries.append((index, len(tokenized), prompt))
    return exceeding_entries


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def print_device_info():
    print(f"Using {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print_device_info()

combined_data = pd.read_csv('eu_combined_final_skrajsan_engleski.csv')


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
combined_data = combined_data.dropna(subset=['Age_Group'])  # Drop rows where age group is not defined


label_encoder = LabelEncoder()
#combined_data['Stance'] = combined_data['Stance'].replace({'NEUTRALNO': 'NEUTRALNO', 'ZA': 'ZA', 'PROTIV': 'PROTIV'})
combined_data['Stance'] = combined_data['Stance'].replace({'NEUTRAL': 'NEUTRAL', 'FOR': 'FOR', 'AGAINST': 'AGAINST'})
combined_data['labels'] = label_encoder.fit_transform(combined_data['Stance'])
joblib.dump(label_encoder, 'label_encoder.joblib')

combined_data['Speaker_party_name'] = combined_data['Speaker_party_name'].replace('-', 'Ostalo')

undefined_gender = combined_data[combined_data['Speaker_gender'] == 'U']
if not undefined_gender.empty:
    print("Rows with undefined gender (U):")
    print(undefined_gender)

combined_data = combined_data[combined_data['Speaker_gender'].isin(['M', 'F'])]

model_name = "launch/POLITICS"
tokenizer = AutoTokenizer.from_pretrained(model_name)

exceeding_entries = check_token_lengths(combined_data, tokenizer)

if exceeding_entries:
    print(f"Found {len(exceeding_entries)} entries exceeding {tokenizer.model_max_length} tokens:")
    for index, length, prompt in exceeding_entries:
        print(f"Index {index} exceeds with {length} tokens. Prompt: {prompt[:100]}...")
else:
    print("No entries exceed the tokenization limit.")



first_input = combined_data['Relevant_Speech'].iloc[0]

politics_tokenized = tokenizer(first_input, truncation=False)

politics_tokens = tokenizer.convert_ids_to_tokens(politics_tokenized['input_ids'])

print("POLITICS Tokenizer Output:")
print("Token IDs:", politics_tokenized['input_ids'])
print("Tokens:", politics_tokens)



n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_predictions = []
all_true_labels = []
metrics_per_fold = []

# K-fold Cross Validation
for fold, (train_idx, test_idx) in enumerate(skf.split(combined_data, combined_data['labels'])):
    print(f"Fold {fold + 1}/{n_splits}")
    
   
    train_data = combined_data.iloc[train_idx]
    test_data = combined_data.iloc[test_idx]


    train_data = oversample_classes(train_data, label_encoder, target_class='NEUTRAL')

    duplicate_speeches = train_data['Relevant_Speech'].value_counts()
    duplicates = duplicate_speeches[duplicate_speeches > 1]

    if not duplicates.empty:
        print(f"Fold {fold + 1}: Duplicates found in the training set based on 'Relevant_Speech'!")
        for speech, count in duplicates.items():
            duplicate_rows = train_data[train_data['Relevant_Speech'] == speech]
            print(f"Speech: '{speech}' - Count of duplicates: {count}")
            print(duplicate_rows)
    else:
        print(f"Fold {fold + 1}: No duplicates in the training set based on 'Relevant_Speech'.")

    train_speeches = set(train_data['Relevant_Speech'])
    test_speeches = set(test_data['Relevant_Speech'])
    overlaps = train_speeches.intersection(test_speeches)

    if overlaps:
        print(f"Overlap found in fold {fold + 1}: {len(overlaps)} overlapping speeches.")
        print(overlaps)
    else: 
        print("No overlaps found")

    
    filename = f'fold_{fold + 1}_samples.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Fold {fold + 1}/10\n")
        
        
        f.write("Train examples:\n")
        for i, (idx, row) in enumerate(train_data.iterrows(), 1):
            f.write(f"{i}. Index {idx} - {row['Relevant_Speech']} - {row['Stance']}\n")
        
        
        f.write("\nTest examples:\n")
        for i, (idx, row) in enumerate(test_data.iterrows(), 1):
            f.write(f"{i}. Index {idx} - {row['Relevant_Speech']} - {row['Stance']}\n")

    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

   
    train_dataset = train_dataset.remove_columns('__index_level_0__') if '__index_level_0__' in train_dataset.column_names else train_dataset
    test_dataset = test_dataset.remove_columns('__index_level_0__') if '__index_level_0__' in test_dataset.column_names else test_dataset

    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})


    def construct_prompt(examples):
        prompts = []
        for i in range(len(examples['Relevant_Speech'])):
            prompt = examples['Relevant_Speech'][i] + "[SEP]Stance towards Europian Union (EU), joining the Europian union and generally europian vales is  [MASK]."
            
            prompts.append(prompt)
        examples['prompt'] = prompts
        return examples

    datasets = datasets.map(construct_prompt, batched=True)

    model_name = "launch/POLITICS"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_), device_map='balanced')

    tokenizer.pad_token = tokenizer.eos_token


    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=512)
        tokenized_inputs["labels"] = examples["labels"]
        return tokenized_inputs

    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["prompt"])

    tokenized_datasets = tokenized_datasets.remove_columns(["Speaker_birth", "Speaker_ID", "Speaker_name", "Speaker_party_name", "Speaker_gender", "Stance", "Relevant_Speech", "Age_Group"])

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

    eval_results = trainer.evaluate()
    print(f"Evaluation Results for fold {fold + 1}:", eval_results)

    model.eval()  
    predictions = trainer.predict(tokenized_datasets['test'])
    predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

    all_predictions.extend(predicted_labels)
    all_true_labels.extend(test_data['labels'].values)

    true_labels = test_data['labels'].values
    predicted_stances = label_encoder.inverse_transform(predicted_labels)
    true_stances = label_encoder.inverse_transform(true_labels)

    target_names = list(label_encoder.classes_)

    precision, recall, f1, _ = precision_recall_fscore_support(true_stances, predicted_stances, average='weighted')
    accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
    metrics_per_fold.append({'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy})

    report = classification_report(true_stances, predicted_stances, target_names=target_names)
    print(f"Classification report for fold {fold + 1}:\n", report)


metrics_df = pd.DataFrame(metrics_per_fold)
metrics_variance = metrics_df.var()
print(f"Variance of metrics across folds:\n{metrics_variance}")


precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))

print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")
print(f"Overall F1-Score: {f1}")
print(f"Overall Accuracy: {accuracy}")


combined_data['predicted_stances'] = label_encoder.inverse_transform(all_predictions)

def map_party_to_group(party_name):
    for group, parties in ideological_groups.items():
        if party_name in parties:
            return group
    return "Unknown"


combined_data['Ideological_Group'] = combined_data['Speaker_party_name'].apply(map_party_to_group)
