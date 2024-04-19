from datasets import load_dataset
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Load the dataset
dataset = load_dataset("midas/duc2001", "raw")["test"]

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Constants
MAX_LEN = 75
BATCH_SIZE = 32

# Prepare mapping for labels
tag2idx = {'B': 0, 'I': 1, 'O': 2}

# Adjust these weights based on your specific dataset and class imbalance
class_weights = torch.tensor([10.0, 15.0, 0.1])  # Example weights for 'B', 'I', 'O'
# class_weights = torch.tensor([10.0, 15.0, 0.1]).cuda()  # Example weights for 'B', 'I', 'O' if GPU applicable

# Tokenization and encoding for BERT
input_ids = []
attention_masks = []
labels = []

for i, item in enumerate(dataset):
    # Join tokens into a single string
    text = ' '.join([t.lower() for t in item['document']])
    tags = item['doc_bio_tags']

    # Encode text
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Prepare labels
    tag_ids = [tag2idx[tag] for tag in tags] + [tag2idx['O']] * (MAX_LEN - len(tags))
    tag_ids = tag_ids[:MAX_LEN]  # Ensure label length matches input length

    input_ids.append(encoded_dict['input_ids'][0])
    attention_masks.append(encoded_dict['attention_mask'][0])
    labels.append(torch.tensor(tag_ids))

# Convert lists to tensors
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
labels = torch.stack(labels)

# Split into training and validation sets
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(
    input_ids, labels, attention_masks, test_size=0.1, random_state=2018
)

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
valid_data = TensorDataset(val_inputs, val_masks, val_labels)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

# Load BERT for token classification
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(tag2idx),
    output_attentions=False,
    output_hidden_states=False,
)

# Set up the optimizer
# optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8)  # increased learning rate


# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate the accuracy of predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# We have a class imbalance which is hindering our model performance
# Apply focal loss to focus more on hard-to-classify examples by down-weighting the loss contributed by well-classified examples(easy-classify)
def hybrid_loss(logits, labels, weights, alpha=0.8, gamma=2.0):
    # Softmax and cross entropy loss
    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none', weight=weights)
    
    # Calculate probabilities of the true class
    p_t = torch.exp(-ce_loss)
    
    # Calculate focal component
    focal_loss = (alpha * (1 - p_t) ** gamma * ce_loss).mean()
    
    return focal_loss

# Training loop
for epoch in tqdm(range(5), desc="Epoch"):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        model.zero_grad()
        
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = hybrid_loss(outputs.logits.view(-1, 3), b_labels.view(-1), class_weights)

        # # Apply class weights
        # log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        # weighted_loss = torch.nn.functional.nll_loss(log_probs.view(-1, model.num_labels), b_labels.view(-1), weight=class_weights)

        # weighted_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # total_loss += weighted_loss.item()
        total_loss += loss.item()


    print(f'Epoch {epoch+1}: Average Training Loss: {total_loss / len(train_dataloader):.2f}')

    # Validation step
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    
    print(f'Validation Accuracy: {eval_accuracy / nb_eval_steps:.2f}')

# Save the model and the tokenizer
model.save_pretrained('./model_save/')
tokenizer.save_pretrained('./model_save/')

# Load the model and the tokenizer
model = BertForTokenClassification.from_pretrained('./model_save/')
tokenizer = BertTokenizer.from_pretrained('./model_save/')

def keywordextract(text, model, tokenizer, device):
    # Tokenize input
    encoded_dict = tokenizer.encode_plus(
        text,                      # Document to encode.
        add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
        max_length=64,             # Pad or truncate.
        padding='max_length',      # Pad to max_length.
        truncation=True,           # Truncate to max_length.
        return_attention_mask=True,# Construct attention masks.
        return_tensors='pt',       # Return PyTorch tensors.
    )
    
    # Move tensors to the correct device
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Decode predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions[0].tolist()  # Remove the batch dimension and convert to list

    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # print("Tokens and Predictions:")  # Debugging output
    # for token, prediction in zip(tokens, predictions):
    #     print(f"{token}: {prediction}")

    # Extract keywords based on the 'B' and 'I' predictions
    keywords = []
    current_keyword = []
    for token, pred in zip(tokens, predictions):
        if pred == 1:  # Corresponds to 'B'
            if current_keyword:  # Save the previous keyword if it exists
                keywords.append("".join(current_keyword).replace("##", ""))
            current_keyword = [token]  # Start a new keyword
        elif pred == 2 and current_keyword:  # Corresponds to 'I'
            current_keyword.append(token)
        else:
            if current_keyword:
                keywords.append("".join(current_keyword).replace("##", ""))
                current_keyword = []

    # Check if the last token was part of a keyword
    if current_keyword:
        keywords.append("".join(current_keyword).replace("##", ""))

    return keywords

text = """Machine learning (ML) is a field of study in artificial intelligence 
concerned with the development and study of statistical algorithms that 
can learn from data and generalize to unseen data, and thus 
perform tasks without explicit instructions."""
keywords = keywordextract(text, model, tokenizer, device)
print("Extracted Keywords:", keywords)


# other embedding transformation
# something not zero padding