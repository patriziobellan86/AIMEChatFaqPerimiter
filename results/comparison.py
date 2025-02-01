import json
import os
import pandas as pd

files = ['bart.json', 'roberta.json', 'deberta.json', 'baseline.json',
         'gold_answer.json',
         'Cognitive Load Theory-1.2-0.9-1-Llama-3.1-70B-Instruct_results_exp1.json',
         'Cognitive Load Theory-1.2-0.9-1-Llama-3.1-70B-Instruct_results_exp2.json',
         'Cognitive Load Theory-1.2-0.9-1-Llama-3.1-70B-Instruct_results_exp3.json',
         'Cognitive Load Theory-1.2-0.9-1-Llama-3.1-70B-Instruct_results_finaldecisor.json',
         'Cognitive Load Theory-1.2-0.9-1-Llama-3.1-70B-Instruct_results_majorityvote.json',

         'User Design Persona-1.2-0.9-1-Llama-3.1-70B-Instruct_results_exp1.json',
         'User Design Persona-1.2-0.9-1-Llama-3.1-70B-Instruct_results_exp2.json',
         'User Design Persona-1.2-0.9-1-Llama-3.1-70B-Instruct_results_exp3.json',
         'User Design Persona-1.2-0.9-1-Llama-3.1-70B-Instruct_results_finaldecisor.json',
         'User Design Persona-1.2-0.9-1-Llama-3.1-70B-Instruct_results_majorityvote.json',

         ]

# Read each JSON file and store data in a dictionary
data_dicts = {}
for file in files:
    with open(file, 'r') as f:
        json_data = json.load(f)
        # Use the filename without extension as the key for the dictionary
        data_dicts[os.path.splitext(os.path.basename(file))[0]] = json_data

# Find common IDs across all files
common_ids = set(data_dicts[next(iter(data_dicts))].keys())  # start with keys from the first file
for data in data_dicts.values():
    common_ids.intersection_update(data.keys())

# Filter data to include only common IDs and create a DataFrame
filtered_data = {}
for file_name, data in data_dicts.items():
    filtered_data[file_name] = {id: data[id] for id in common_ids}

# Convert to DataFrame
df = pd.DataFrame(filtered_data)

# Display the DataFrame
print(df)


# Function to calculate accuracy compared to 'gold_answer'
def calculate_accuracy(df, reference):
    accuracies = {}
    for col in df.columns:
        if col != reference:
            # Calculate the proportion of matching entries
            accuracy = (df[col] == df[reference]).mean()
            accuracies[col] = accuracy
    return accuracies


# Call the function and print the accuracy scores
accuracy_scores = calculate_accuracy(df, 'gold_answer')
print(accuracy_scores)

from pprint import pprint

pprint(accuracy_scores)

print(sorted(accuracy_scores, key=accuracy_scores.get, reverse=True))

#  save accuracy scores ordered starting from the best scorer in a csv file
df_accuracy = pd.DataFrame(accuracy_scores, index=['accuracy'])
df_accuracy = df_accuracy.T
df_accuracy = df_accuracy.sort_values(by='accuracy', ascending=False)
df_accuracy.to_csv('accuracy_scores.csv', sep=';')
