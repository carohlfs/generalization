import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()

# Sentence with the word "new" masked
sentence = "Four score and seven years ago our fathers brought forth on this continent, a [MASK] nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors='pt')
inputs = inputs.to(device)
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# Ensure the input contains a [MASK] token
if mask_token_index.size(0) == 0:
    raise ValueError("No [MASK] token found in the input.")

# Predict the masked word
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits for the masked token
logits = outputs.logits[0, mask_token_index, :]

# Get the probabilities and top 5 predictions
probs = torch.softmax(logits, dim=-1).squeeze()
top_probs, top_indices = torch.topk(probs, 5)
top_words = [tokenizer.decode(idx) for idx in top_indices]

# Print the top 5 predictions with their probabilities
top_predictions = list(zip(top_words, top_probs.tolist()))
for word, prob in top_predictions:
    print(f"Word: {word}, Probability: {prob}")

# Define a function to compute gradients with respect to embeddings
def compute_gradients(input_ids, attention_mask, target_index):
    inputs_embeds = model.bert.embeddings(input_ids).clone().detach().to(device).requires_grad_(True)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits[0, mask_token_index, :]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_index], device=device))
    loss.backward()
    return inputs_embeds.grad

# Prepare the inputs
input_ids = inputs['input_ids'].clone().detach().to(device).long()
attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

# Compute Shapley values for the top 5 predictions
shap_values_matrix = []

for predicted_token_id in top_indices:
    predicted_token_id = predicted_token_id.item()  # Ensure scalar conversion
    print(f"predicted_token_id: {predicted_token_id}")  # Debugging
    try:
        gradients = compute_gradients(input_ids, attention_mask, predicted_token_id)
        shap_values_matrix.append(gradients.squeeze().detach().cpu().numpy())
    except (IndexError, ValueError) as e:
        print(f"Error: {e}")  # Debugging
        continue

# Convert Shapley values to a NumPy array for easier manipulation
shap_values_matrix = np.array(shap_values_matrix)

# Reduce the hidden state dimension by averaging across it
shap_values_reduced = shap_values_matrix.mean(axis=2)  # You can also use sum(axis=2) if preferred

# Tokenize the sentence to get individual words
tokenized_sentence = tokenizer.convert_ids_to_tokens(input_ids[0])

# Create a DataFrame for the reduced Shapley values matrix
shap_df = pd.DataFrame(shap_values_reduced, index=top_words, columns=tokenized_sentence)

# Transpose the DataFrame
shap_values_df = shap_df.transpose()

# Remove the [MASK] entry
# shap_values_df = shap_values_df.drop('[MASK]', errors='ignore')

# Save the transposed DataFrame to a text file
shap_values_df.to_csv("shap_values_df.txt", sep='\t')

# Print the transposed DataFrame to verify
print(shap_values_df)

# Extract SHAP values for the top 5 predictions
shap_great = shap_values_df['g r e a t']
shap_new = shap_values_df['n e w']
shap_united = shap_values_df['u n i t e d']
shap_free = shap_values_df['f r e e']
shap_christian = shap_values_df['c h r i s t i a n']

# Scale the SHAP values for better visualization
scaling_factor = 1e10
shap_great_scaled = shap_great * scaling_factor
shap_new_scaled = shap_new * scaling_factor
shap_united_scaled = shap_united * scaling_factor
shap_free_scaled = shap_free * scaling_factor
shap_christian_scaled = shap_christian * scaling_factor

# Create a DataFrame with the scaled SHAP values, in the reverse order
shap_combined_df = pd.DataFrame({
    'Christian': shap_christian_scaled,
    'Free': shap_free_scaled,
    'United': shap_united_scaled,
    'New': shap_new_scaled,
    'Great': shap_great_scaled
})

# Ensure the words in the sentence appear from top to bottom in their original order
shap_combined_df = shap_combined_df.iloc[::-1]

# Plot the grouped bar graph with words in the correct order
fig, ax = plt.subplots(figsize=(7.45, 10))  # Adjusted width to be slightly wider
shap_combined_df.plot(kind='barh', ax=ax, color=['orange', 'purple', 'red', 'green', 'blue'], width=0.9)

ax.set_xlabel('Scaled SHAP Value', fontsize=18)
ax.set_ylabel('Word in Sentence', fontsize=18)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)

# Add light dotted horizontal black lines in between each word
for i in range(1, len(shap_combined_df) + 1):
    ax.axhline(i - 0.5, color='black', linestyle=':', linewidth=0.5)

# Fix the legend to show correct colors and order
handles, labels = ax.get_legend_handles_labels()
order = [4, 3, 2, 1, 0]  # The desired order: blue (Great) to orange (Christian)
legend = ax.legend([handles[idx] for idx in order], ['Great', 'New', 'United', 'Free', 'Christian'], fontsize=18, loc='upper left', bbox_to_anchor=(-0.015, 0.98))  # Adjusted legend position

# Set the legend background to white and make the edges black
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

# Add the legend as an artist
plt.gca().add_artist(legend)

plt.tight_layout()
plt.savefig('gettysburg_shapley.png')
plt.show()

# Define the predicted words and their probabilities
predicted_words = ['Great', 'New', 'United', 'Free', 'Christian']
probabilities = [top_probs[0].item(), top_probs[1].item(), top_probs[2].item(), top_probs[3].item(), top_probs[4].item()]

# Convert probabilities to percentages
probabilities_percentage = [prob * 100 for prob in probabilities]

# Create a DataFrame for the bar graph
prob_df = pd.DataFrame({'Word': predicted_words, 'Probability': probabilities_percentage})

# Plot the bar graph
fig, ax = plt.subplots(figsize=(10, 3))

# Define the colors for the bars
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Plot the bars
bars = ax.bar(prob_df['Word'], prob_df['Probability'], color=colors)

# Set the font sizes for the axes and labels
ax.set_ylabel('Probability', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Set the y-axis ticks and labels
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels(['0%', '5%', '10%', '15%'])

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords='offset points',
                ha='center', fontsize=24)

# Cut the horizontal axis title
ax.set_xlabel('')

# Ensure the vertical axis is visible
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)

plt.tight_layout()
plt.savefig('gettysburg_probs.png')
plt.show()