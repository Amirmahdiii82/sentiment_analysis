import torch
import torch.nn as nn
from transformers import BertTokenizerFast
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

tokenizer = BertTokenizerFast.from_pretrained("HooshvareLab/bert-fa-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PersianTransformerSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, n_heads=8, n_layers=4, hidden_dim=512, output_dim=2, max_seq_length=128, dropout=0.2, pad_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_length, embedding_dim))

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, text, attention_mask=None):
        x = self.embedding(text) + self.pos_embedding[:text.size(1), :]
        x = self.layer_norm(x)

        if attention_mask is not None:
            attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0.0)

        transformer_output = self.transformer(x, src_key_padding_mask=attention_mask)
        pooled = transformer_output.mean(dim=1)

        x = self.fc1(pooled)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        output = self.fc3(x)
        return output

model = PersianTransformerSentiment(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=256,
    hidden_dim=512,
    output_dim=2,
    max_seq_length=128,
    pad_idx=tokenizer.pad_token_id
)
model.load_state_dict(torch.load("analysis/ml_model/model_best.pt", map_location=device))
model.to(device)
model.eval()

def generate_prediction(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()

    sentiment_map = {0: 'HAPPY', 1: 'SAD'}
    return sentiment_map.get(predicted_class, "Unknown")
