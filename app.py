import streamlit as st
import torch
import torch.nn     as nn
import torch.nn.functional as F
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unidecode
import re


idx2label = {0:'neutral', 1:'negative', 2:'positive'}

def preprocess_text(text):
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    text = unidecode.unidecode(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords_set]
    text = ' '.join(words)
    return text

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,n_layers,n_classes,dropout_prob):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_size,n_layers,batch_first=True,bidirectional=True)
        self.norm=nn.LayerNorm(hidden_size * 2) # Adjusted for bidirectional LSTM output
        self.dropout=nn.Dropout(dropout_prob)
        self.fc1=nn.Linear(hidden_size * 2,16) # Adjusted for bidirectional LSTM output
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(16,n_classes)
    def forward(self,x):
      x=self.embedding(x)
      output, (hidden, cell) = self.lstm(x) # Unpack the tuple
      x,_ = torch.max(output, dim=1)
      x = self.norm(x)
      x=self.dropout(x)
      x=self.fc1(x)
      x=self.relu(x)
      x=self.fc2(x)
      return x

with open('vocab.pkl', 'rb') as f:
    vocab_list=pickle.load(f)
    vocab={word:idx for idx,word in enumerate(vocab_list)}
@st.cache_resource
def load_model(model_path):
    model=SentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_size=64,
        n_layers=1,
        n_classes=3,
        dropout_prob=0.5
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model
def inference(sentence_text, model, max_seq_len, device):
    # Preprocess the input sentence using the same function as for training data
    processed_sentence = preprocess_text(sentence_text)

    # Encode the sentence using vocab
    tokens=processed_sentence.split()
    unk_id = vocab.get("<unk>", 0)
    pad_id = vocab.get("<pad>", 0)
    input_ids = [vocab.get(token, unk_id) for token in tokens]

    # Pad or truncate the input_ids to max_seq_len
    if len(input_ids) < max_seq_len:
        padding_needed = max_seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * padding_needed
    elif len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    # Convert to tensor, add batch dimension, and move to device
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(input_tensor)

    # Apply softmax to get probabilities
    preds = F.softmax(predictions, dim=1)
    p_max, yhat = torch.max(preds.data, 1)

    # Return predicted probability (as percentage) and class label
    predicted_class_prob = p_max.item()
    predicted_class_label = yhat.item()

    return predicted_class_prob * 100, predicted_class_label

model=load_model('bilstm_model.pth')
max_seq_len=32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
  st.title('Sentiment Analysis for Financial News')
  st.title('Model: BiLSTM.')
  text_input = st.text_input("Sentence: ", "The price of this coin is increasing significantly")
  p, idx = inference(text_input, model, max_seq_len, device)
  label = idx2label[idx]
  st.success(f'Sentiment: {label} with {p:.2f} % probability.') 

if __name__ == '__main__':
     main() 
