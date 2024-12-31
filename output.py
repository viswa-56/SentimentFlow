import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

# Define the BERTForClassification class
class BERTForClassification(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.attention_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)

        # Calculate attention weights
        attention_scores = self.attention_layer(outputs.last_hidden_state).squeeze(-1)
        attention_weights = self.sigmoid(attention_scores)

        return logits, attention_weights

# Load the tokenizer and BERT model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# Initialize the classifier
classifier = BERTForClassification(bert_model, num_classes=2)

# Load the model's state dictionary
model_save_path = './static/bert_classifier_state_dict.pth'
classifier.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
classifier.eval()

# Function to predict the label for input text
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        logits, _ = classifier(input_ids, attention_mask, token_type_ids)
        predicted_label = torch.argmax(logits, dim=1).item()
    if predicted_label==0:
        val="Negative Sentiment"
    else:
        val="Positive Sentiment"
    return predicted_label,val




#
#
# def get_output(text):
#     encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     input_ids = encoded_input['input_ids']
#     attention_mask = encoded_input['attention_mask']
#     token_type_ids = encoded_input['token_type_ids']
#     logits, attention_weights = __model(input_ids, attention_mask, token_type_ids)
#     probabilities = torch.softmax(logits, dim=1)
#
# def load_saved_artifacts():
#     global __model
#     print("loading artifacts...")
#     with open("./static/bert_classifier.pkl", 'rb') as f:
#         __model = pickle.load(f)
#     print("loading artifacts is done...")


if __name__ == "__main__":
    input_text = "The movie was fantastic!"
    predicted_label = predict(input_text)
    print(f'Predicted label: {predicted_label}')
