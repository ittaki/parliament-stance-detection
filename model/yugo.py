import os
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, EarlyStoppingCallback, AutoModelForSequenceClassification 
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
import joblib
from imblearn.over_sampling import RandomOverSampler
from huggingface_hub import login
import numpy as np


hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def print_device_info():
    print(f"Using {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


print_device_info()

combined_data = pd.read_csv('eu_combined_final_skrajsan.csv')

label_encoder = LabelEncoder()
combined_data['Stance'] = combined_data['Stance'].replace({'NEUTRALNO': 'NEUTRALNO', 'ZA': 'ZA', 'PROTIV': 'PROTIV'})
combined_data['labels'] = label_encoder.fit_transform(combined_data['Stance'])
joblib.dump(label_encoder, 'label_encoder.joblib')


combined_data['Speaker_party_name'] = combined_data['Speaker_party_name'].replace('-', 'Ostalo')

undefined_gender = combined_data[combined_data['Speaker_gender'] == 'U']
if not undefined_gender.empty:
    print("Rows with undefined gender (U):")
    print(undefined_gender)


combined_data = combined_data[combined_data['Speaker_gender'].isin(['M', 'F'])]


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
        prompt = f"Text: {row['Relevant_Speech']} Stance:"
        tokenized = tokenizer.encode(prompt, truncation=False)
        if len(tokenized) > max_length:
            exceeding_entries.append((index, len(tokenized), prompt))
    return exceeding_entries


model_name = "gordicaleksa/YugoGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)


print("Class distribution before oversampling:")
print(combined_data['Stance'].value_counts())



n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_predictions = []
all_true_labels = []
all_relevant_speeches = []
metrics_per_fold = []

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_), device_map='balanced')

tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

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


   
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

   
    if '__index_level_0__' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns('__index_level_0__')
    if '__index_level_0__' in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns('__index_level_0__')

    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

   
    def construct_prompt(examples):
        prompts = []
        for i in range(len(examples['Relevant_Speech'])):
            prompt = f"[INST] Odredi stav sledećeg govora prema Evropi, pridruživanju Evropskoj uniji ili evropskim vrednostima je: '{examples['Relevant_Speech'][i]}' [/INST]"
   
            prompts.append(prompt)
        examples['prompt'] = prompts
        return examples

    datasets = datasets.map(construct_prompt, batched=True)


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


    tokenized_datasets = tokenized_datasets.remove_columns(["Speaker_birth", "Speaker_ID", "Speaker_name", "Speaker_party_name", "Speaker_gender", "Stance", "Relevant_Speech"])

    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold + 1}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=True,
        save_total_limit=1,
        remove_unused_columns=False,
        metric_for_best_model='eval_loss', 
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
    all_relevant_speeches.extend(test_data['Relevant_Speech'].values) 


    true_labels = test_data['labels'].values
    predicted_stances = label_encoder.inverse_transform(predicted_labels)
    true_stances = label_encoder.inverse_transform(true_labels)

    target_names = list(label_encoder.classes_)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    accuracy = np.mean(predicted_labels == true_labels)
    metrics_per_fold.append({'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy})

    report = classification_report(true_stances, predicted_stances, target_names=target_names)
    print(f"Classification report for fold {fold + 1}:\n", report)


precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))

print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")
print(f"Overall F1-Score: {f1}")
print(f"Overall Accuracy: {accuracy}")


metrics_df = pd.DataFrame(metrics_per_fold)
variance = metrics_df.var()
print(f"Variance of metrics across folds:\n{variance}")


results_df = pd.DataFrame({
    'Relevant_Speech': all_relevant_speeches,
    'True_Stance': label_encoder.inverse_transform(all_true_labels),
    'Predicted_Stance': label_encoder.inverse_transform(all_predictions)
})


results_df.to_csv('/shared/home/andela.rajovic/eu/gpt/model_predictions_analysis.csv', index=False)
