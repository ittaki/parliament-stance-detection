import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from huggingface_hub import login
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from transformers import DataCollatorWithPadding

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

combined_data = pd.read_csv('LEX_zdruzeni_skrajsan.csv')

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

label_encoder = LabelEncoder()
combined_data['Stance'] = combined_data['Stance'].replace({'NEUTRALNO': 'NEUTRALNO', 'ZA': 'ZA', 'PROTIV': 'PROTIV'})
combined_data['labels'] = label_encoder.fit_transform(combined_data['Stance'])
joblib.dump(label_encoder, 'label_encoder.joblib')


combined_data['Speaker_party_name'] = combined_data['Speaker_party_name'].replace('-', 'Ostalo')


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

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)


exceeding_entries = check_token_lengths(combined_data, tokenizer)
if exceeding_entries:
    print(f"Found {len(exceeding_entries)} entries exceeding {tokenizer.model_max_length} tokens:")
    for index, length, prompt in exceeding_entries:
        print(f"Index {index} exceeds with {length} tokens. Prompt: {prompt[:100]}...")
else:
    print("No entries exceed the tokenization limit.")


first_input = combined_data['Relevant_Speech'].iloc[0]
llama_tokenized = tokenizer(first_input, truncation=False)
llama_tokens = tokenizer.convert_ids_to_tokens(llama_tokenized['input_ids'])

print("Llama Tokenizer Output:")
print("Token IDs:", llama_tokenized['input_ids'])
print("Tokens:", llama_tokens)

print("Class distribution before oversampling:")
print(combined_data['Stance'].value_counts())

print("Class distribution after oversampling:")
print(combined_data['Stance'].value_counts())


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


all_predictions = []
all_true_labels = []
all_relevant_speeches = []
metrics_per_fold = []
all_indices = [] 


local_model_dir = '/shared/home/andela.rajovic/eu/llama/Meta-Llama-3.1-8B-Instruct'

if not os.path.exists(local_model_dir):
    print("Loading model from Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token, num_labels=len(label_encoder.classes_), device_map='balanced')
    print(f"Saving model to {local_model_dir}...")
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
else:
    print(f"Loading model from local directory {local_model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir, num_labels=len(label_encoder.classes_), device_map='balanced')
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)


tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

# LORA
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, config)
model.gradient_checkpointing_enable()

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(combined_data, combined_data['labels'])):
    print(f"Fold {fold + 1}/{n_splits}")
    
    train_data = combined_data.iloc[train_idx]
    test_data = combined_data.iloc[test_idx]

    train_data = oversample_classes(train_data, label_encoder, target_class='NEUTRALNO')

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

    train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_data.reset_index(drop=True))
    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})


    def construct_prompt(examples):
        sys_instruction = "<<SYS>>\nVi ste AI asistent koji je podešen za detekciju stava (stance detection) u parlamentarnim govorima.\n<</SYS>>\n"
        prompts = []
        for i in range(len(examples['Relevant_Speech'])):
            #user_prompt = f"[INST] Odredi stav sledećeg govora prema Evropi, pridruživanju Evropskoj uniji ili evropskim vrednostima je: '{examples['Relevant_Speech'][i]}' [/INST]"
            #user_prompt = f"[INST] Stav sledećeg govora prema ilegalnim migrantima, pomaganju migrantima ili integrisanju migranata u drustvo je: '{examples['Relevant_Speech'][i]}' [/INST]"
            user_prompt = f"[INST] Stav sledećeg govora prema upotrebi lex specialis zakonima ili posebnim zakonima je: '{examples['Relevant_Speech'][i]}' [/INST]"
            #user_prompt = f"[INST] Stav sledećeg govora prema organizaciji NATO/pridruživanju Srbije NATO paktu je : '{examples['Relevant_Speech'][i]}' [/INST]"
            assistant_completion = " Stav: "
            prompt = sys_instruction + user_prompt + assistant_completion
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

    tokenized_datasets = tokenized_datasets.remove_columns(["Speaker_birth", "Speaker_ID", "Speaker_name", "Speaker_party_name", "Speaker_gender", "Stance", "Relevant_Speech", "Age_Group"])

    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold + 1}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
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
    all_indices.extend(test_idx)

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



trainer.save_model('/shared/home/andela.rajovic/eu/llama/llama-EU-finetuned')


# izracunaj confusion matrix:

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

plt.savefig('/shared/home/andela.rajovic/eu/llama/confusion_matrix_eu_slovenian.png')



try:

    combined_data.loc[all_indices, 'predicted_stances'] = label_encoder.inverse_transform(all_predictions)
except KeyError as e:

    print(f"Encountered KeyError: {e}. Adjusting indices...")
    
    valid_indices = [idx for idx in all_indices if idx in combined_data.index]
    valid_predictions = [all_predictions[i] for i in range(len(all_indices)) if all_indices[i] in valid_indices]
    
    combined_data.loc[valid_indices, 'predicted_stances'] = label_encoder.inverse_transform(valid_predictions)



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



def plot_stance_distribution_party(data, output_path):
    if data.empty:
        print("No data available for plotting")
        return

    
    grouped_ideology = data.groupby(['Ideological_Group', 'predicted_stances']).size().unstack(fill_value=0)
    grouped_ideology = grouped_ideology[grouped_ideology.sum(axis=1) > 0]
    grouped_ideology_percentage = grouped_ideology.div(grouped_ideology.sum(axis=1), axis=0) * 100
    ordered_groups = ['Skrajno levi', 'Srednje levi', 'Srednje desni', 'Skrajno desni', 'Ostalo']
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

plot_stance_distribution_party(combined_data, '/shared/home/andela.rajovic/eu/llama/stance_distribution_by_party_grouped.png')




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

plot_gender_distribution_separate_bars(combined_data, '/shared/home/andela.rajovic/eu/llama/stance_distribution_by_gender_separate_bars.png')


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

plot_age_group_distribution_separate_bars(combined_data, '/shared/home/andela.rajovic/eu/llama/stance_distribution_by_age_group_separate_bars.png')
