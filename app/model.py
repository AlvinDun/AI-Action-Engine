import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, kernel_sizes=(3,4,5), num_filters=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters*len(kernel_sizes), num_classes)

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embedding(input_ids)            # (B, L, E)
        x = x.transpose(1, 2)                    # (B, E, L)
        xs = []
        for conv in self.convs:
            c = torch.relu(conv(x))              # (B, F, Lk)
            p = torch.max(c, dim=2).values       # (B, F)
            xs.append(p)
        h = torch.cat(xs, dim=1)                 # (B, F*len(K))
        h = self.dropout(h)
        logits = self.fc(h)                      # (B, C)
        return logits
