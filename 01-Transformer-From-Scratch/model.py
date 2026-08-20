import torch
import torch.nn as nn
import math


DEBUG = False

# ==================================================
# Input Embeddings
# ==================================================

class InputEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

    def forward(self, x):

        return self.embedding(x) * math.sqrt(
            self.d_model
        )


# ==================================================
# Positional Encoding
# ==================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        seq_len,
        dropout
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(
            seq_len,
            d_model
        )

        position = torch.arange(
            0,
            seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)


# ==================================================
# Multi Head Attention
# ==================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        h
    ):
        super().__init__()

        self.d_model = d_model
        self.h = h

        

        assert d_model % h == 0

        self.d_k = d_model // h

        self.w_q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.w_k = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.w_v = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.attention_weights = None   # to save the weights instead of throwing them away

    def forward(self, x,mask=None):

        batch_size = x.shape[0]

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(
            batch_size,
            -1,
            self.h,
            self.d_k
        )

        K = K.view(
            batch_size,
            -1,
            self.h,
            self.d_k
        )

        V = V.view(
            batch_size,
            -1,
            self.h,
            self.d_k
        )

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        
        # ------------------------------------
        # Compute Attention Scores
        # ------------------------------------

        scores = (
        Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        if DEBUG:
            print("\n========================")
            print("Scores BEFORE Mask")
            print("========================")
            print(scores[0,0])

        # ------------------------------------
        # Apply Mask
        # ------------------------------------


        if mask is not None:

           scores = scores.masked_fill(
              mask == 0,
              float("-inf")
        )

        if DEBUG:
            print("\n========================")
            print("Scores AFTER Mask")
            print("========================")
            print(scores[0,0])

        # ------------------------------------
        # Softmax
        # ------------------------------------
        
        weights = torch.softmax(
           scores,
           dim=-1
        )

        self.attention_weights = weights.detach().cpu() # Save the Matrix now every forward pass automatically stores (batch,heads,seq_len, seq_len) inside the module.
        # Why detach()? -> without detach Pytorch keeps the computation graph that means attention graph -> gradient graph -> everything. memory usage becomes larger.

        # detach() -> means dont keep gradients
        # cpu() -> store a CPU copy for analysis




        if DEBUG:
            print("\n========================")
            print("Weights AFTER Softmax")
            print("========================")
            print(weights[0,0])


        

        attention = weights @ V

        attention = attention.transpose(
            1,       
            2
        )

        attention = attention.contiguous().view(
            batch_size,
            -1,
            self.d_model
        )

        output = self.w_o(attention)

        return output


# ==================================================
# Layer Normalization
# ==================================================

class LayerNormalization(nn.Module):

    def __init__(
        self,
        eps=1e-6
    ):
        super().__init__()

        self.eps = eps

        self.alpha = nn.Parameter(
            torch.ones(1)
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        std = x.std(
            dim=-1,
            keepdim=True
        )

        return self.alpha * (
            (x - mean)
            /
            (std + self.eps)
        ) + self.bias


# ==================================================
# Feed Forward Network
# ==================================================

class FeedForwardBlock(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(self, x):

        x = self.linear1(x)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x


# ==================================================
# Residual Connection
# ==================================================

class ResidualConnection(nn.Module):

    def __init__(
        self,
        dropout
    ):
        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = LayerNormalization()

    def forward(
        self,
        x,
        sublayer
    ):

        return x + self.dropout(
            sublayer(
                self.norm(x)
            )
        )


# ==================================================
# Encoder Block
# ==================================================

class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block,
        feed_forward_block,
        dropout
    ):
        super().__init__()

        self.self_attention_block = (
            self_attention_block
        )

        self.feed_forward_block = (
            feed_forward_block
        )

        self.residual_connections = (
            nn.ModuleList(
                [
                    ResidualConnection(
                        dropout
                    ),
                    ResidualConnection(
                        dropout
                    )
                ]
            )
        )

    def forward(self, x):

        x = self.residual_connections[0](
            x,
            lambda x:
            self.self_attention_block(x)
        )

        x = self.residual_connections[1](
            x,
            self.feed_forward_block
        )

        return x


# ==================================================
# Encoder Stack
# ==================================================

class Encoder(nn.Module):

    def __init__(
        self,
        layers
    ):
        super().__init__()

        self.layers = layers

        self.norm = LayerNormalization()

        # Store hidden states from every layer
        self.hidden_states = []

    def forward(self, x):

        # clear previous forward pass
        self.hidden_states = []


        for i, layer in enumerate(self.layers):

            x = layer(x)

            # Save hidden state 
            self.hidden_states.append(
                x.detach().cpu()   # x.detach()-> # "Forget how this tensor was created. I only care about its values."

                # why .cpu()? Matplotlib cannot directly plot GPU tensors.So we move them to CPU
            )

            if DEBUG:
                print(
                    f"\nAfter Encoder Layer {i+1}"
                )

                print(
                    x.shape
                )

        return self.norm(x)


# Mask Function

# =====================
# Casual Mask
# =====================

def causal_mask(size):

    mask = torch.tril(
        torch.ones(size,size)

    )

    return mask



# ==================================================
# TEST BLOCK
# ==================================================

if __name__ == "__main__":

    vocab_size = 10000
    d_model = 512

    embedding = InputEmbeddings(
        d_model,
        vocab_size
    )

    positional_encoding = PositionalEncoding(
        d_model=512,
        seq_len=10,
        dropout=0.1
    )

    sample = torch.tensor(
        [
            [10, 25, 13, 8]
        ]
    )

    x = embedding(sample)

    if DEBUG:
        print("\nEmbedding Shape:")
        print(x.shape)

    x = positional_encoding(x)

    if DEBUG:
        print("\nAfter Positional Encoding:")
        print(x.shape)

    # ------------------------------------
    # Encoder Layers
    # ------------------------------------

    encoder_layers = nn.ModuleList(

        [
            EncoderBlock(
                MultiHeadAttention(
                    d_model=512,
                    h=8
                ),
                FeedForwardBlock(
                    d_model=512,
                    d_ff=2048,
                    dropout=0.1
                ),
                dropout=0.1
            )

            for _ in range(6)
        ]

    )

    encoder = Encoder(
        encoder_layers
    )

    encoder_output = encoder(x)

    if DEBUG:
        print("\n========================")
        print("MASKED ATTENTION TEST")
        print("========================")

    mask = causal_mask(4)

    multi_head = MultiHeadAttention(
        d_model=512,
        h=8
    )

    output, weights = multi_head(
        x,
        mask
    )

    if DEBUG:
        print("\nOutput Shape")
        print(output.shape)

    if DEBUG:
        print("\nWeights Shape")
        print(weights.shape)


    if DEBUG:
        print(
            "\nFinal Encoder Output Shape:"
        )

        print(
            encoder_output.shape
        )

        print(
            "\nFirst Token First 10 Features:"
        )

        print(
            encoder_output[0,0,:10]
        )


