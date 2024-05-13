import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,x_dim, y_dim, z_dim, x_heads, y_heads, z_heads):
        super(MultiHeadAttention, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.x_heads = x_heads
        self.y_heads = y_heads
        self.z_heads = z_heads
        self.x_head_dim = x_dim // x_heads
        self.y_head_dim = y_dim // y_heads
        self.z_head_dim = z_dim // z_heads

        # Linear layers for queries, keys, and values for each head
        self.linear_query = nn.Linear(x_dim, self.x_head_dim)
        self.linear_key = nn.Linear(y_dim, self.y_head_dim * y_heads)
        self.linear_value = nn.Linear(z_dim, self.z_head_dim * z_heads)

    def forward(self,x,y,z):
        # Linear transformation for queries, keys, and values for each head
        query = self.linear_query(x)
        key = self.linear_key(y)
        value = self.linear_value(z)
        # Reshape queries, keys, and values to split into multiple heads
        query = query.view(self.x_heads, x.size(0)//self.x_heads, self.x_dim//self.x_heads).transpose(1, 2)
        key = key.view(self.y_heads, y.size(0)//self.y_heads, self.y_dim//self.y_heads)
        value = value.view(self.z_heads, z.size(0)//self.z_heads, self.z_dim//self.z_heads)
        # Calculate attention scores
        attn_scores = torch.matmul(query, key) 
        attn_probs = F.softmax(attn_scores.view(attn_scores.size(0) * attn_scores.size(1), -1), dim=-1)
        # Apply attention scores to values
        attn_output = torch.squeeze(torch.matmul(value, attn_probs))
        return attn_output

# Example usage:
modal1_dim = 512
modal2_dim = 512
modal3_dim = 512

num_heads_modal1 = 4
num_heads_modal2 = 1
num_heads_modal3 = 1

# Create an instance of MultiModalMultiHeadAttention
multi_modal_attn = MultiHeadAttention(modal1_dim, modal2_dim, modal3_dim,
                                               num_heads_modal1, num_heads_modal2, num_heads_modal3)

# Generate some random input data for each modality
modal1_data = torch.randn(40, 512)  # Example modal1 input of shape (40, 512)
modal2_data = torch.randn(10, 512)  # Example modal2 input of shape (10, 512)
modal3_data = torch.randn(10, 512)  # Example modal3 input of shape (10, 512)

# Get outputs from MultiModalMultiHeadAttention for each modality
# modal1_output, modal2_output, modal3_output = multi_modal_attn(modal1_data, modal2_data, modal3_data)
output = multi_modal_attn(modal1_data, modal2_data, modal3_data)
# Print shapes of outputs for each modality
print("Output shape:", output.shape)
