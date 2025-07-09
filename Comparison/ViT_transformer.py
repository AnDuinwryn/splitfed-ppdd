import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------#
class PatchEmbedding(nn.Module):
    def __init__(self, in_channel=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        # self.embed_dim = pow(self.patch_size, 2) * in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=self.in_channel, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
#-----------------------------------------------------------------------------------------------------------------------------#
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))  # Adjusted for [CLS] token

    def forward(self, x):
        return x + self.pos_embed
#-----------------------------------------------------------------------------------------------------------------------------#
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attn(x, x, x)[0]
#-----------------------------------------------------------------------------------------------------------------------------#
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
#-----------------------------------------------------------------------------------------------------------------------------#
class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, seq_len=14*14, num_classes=10, embed_dim=768, num_heads=8, depth=6, mlp_dim=1024):
        super().__init__()
        self.patch_embedding = PatchEmbedding(3, patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, seq_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.mlp_head(x[:, 0])
#-----------------------------------------------------------------------------------------------------------------------------#
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

model = VisionTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Train for 5 epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss/len(train_loader)}")

# if __name__ == "__main__":
#     dummy_input = torch.tensor([dataWrap.fetch_mel_dat()[0]['Train'][0]['Feature']], dtype=torch.float32)
#     print(f"tmp_tensor shape: {dummy_input.shape}")
#     # dummy_input = torch.randn(1, 3, 224, 224)
#     embedding = PatchEmbedding()
#     posEncoding = PositionalEncoding(768, 128)
#     result = posEncoding(embedding(dummy_input))
#     print(result.shape)