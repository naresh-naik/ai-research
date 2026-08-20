import torch
import torch.nn as nn
import math

from model import LayerNormalization, ResidualConnection

DEBUG = False

# ==================================================
# Masked Multi Head Attention
# ==================================================

class MaskedMultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        h,
        debug=False
    ):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.debug = debug

        assert d_model % h == 0

        self.d_k = d_model // h

        # Query Projection
        self.w_q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        # Key Projection
        self.w_k = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        # Value Projection
        self.w_v = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        # Output Projection
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def forward(
        self,
        x,
        mask=None
    ):

        batch_size = x.shape[0]

        # ---------------------------------
        # Step 1 : Create Q, K and V
        # ---------------------------------

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # ---------------------------------
        # Step 2 : Split into Heads
        # (batch, seq_len, d_model)
        # ->
        # (batch, seq_len, heads, d_k)
        # ---------------------------------

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

        # ---------------------------------
        # Step 3 : Move Head Dimension
        # (batch, seq_len, heads, d_k)
        # ->
        # (batch, heads, seq_len, d_k)
        # ---------------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ---------------------------------
        # Step 4 : Compute Attention Scores
        # ---------------------------------

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        if self.debug:
            print("\n========================")
            print("SCORES BEFORE MASK")
            print("========================")
            print(scores[0,0])

        # ---------------------------------
        # STEP 5 : Apply Causal Mask
        # ---------------------------------

        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

            if self.debug:
                print("\n========================")
                print("SCORES AFTER MASK")
                print("========================")
                print(scores[0,0])

        # ---------------------------------
        # STEP 6 : Softmax
        # ---------------------------------

        weights = torch.softmax(
            scores,
            dim=-1
        )

        if self.debug:
            print("\n========================")
            print("SOFTMAX WEIGHTS")
            print("========================")
            print(weights[0,0])

        # ---------------------------------
        # STEP 7 : Attention Output
        # ---------------------------------

        attention = weights @ V

        # ---------------------------------
        # STEP 8 : Concatenate Heads
        # ---------------------------------

        attention = attention.transpose(
            1,
            2
        )

        attention = attention.contiguous().view(
            batch_size,
            -1,
            self.d_model
        )

        output = self.w_o(
            attention
        )

        return output, weights



class CrossMultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        h,
        debug=False
    ):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.debug=debug

        assert d_model % h == 0

        self.d_k = d_model // h


        # Projection Layers

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

        # Forward Function

    def forward(
                self,
                query,
                key,
                value,
                mask=None
    ):


            batch_size = query.shape[0]

            Q = self.w_q(query)

            K = self.w_k(key)

            V = self.w_v(value)

    # Split Into Heads

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


        # Move Head Dimension


            Q = Q.transpose(1,2)

            K = K.transpose(1,2)

            V = V.transpose(1,2)


        # Attention Scores


            scores = (
             Q @ K.transpose(-2,-1)
            ) / math.sqrt(self.d_k)


            if self.debug:
                print("\n========================")
                print("CROSS ATTENTION SCORES")
                print("========================")
                print(scores[0,0])


        #  Mask

            if mask is not None:

               scores = scores.masked_fill(
                  mask==0,
                  float("-inf")
            )

        # Softmax

            weights = torch.softmax(
                scores,
                dim=-1
            )


            if self.debug:
                print("\n========================")
                print("CROSS ATTENTION WEIGHTS")
                print("========================")
                print(weights[0,0])

        # Attention Output

            attention = weights @ V

        # Transpose

            attention = attention.transpose(
                1,
                2
            )

        # Concatenate


            attention = attention.contiguous().view(
                batch_size,
                -1,
                self.d_model
            )

        # WO Projection


            output = self.w_o(
                 attention
            )
        # Return

            return output, weights



#  ====================================================
#  Decoder Block
#  ====================================================

# ==================================================
# Decoder Block
# ==================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block,
        cross_attention_block,
        feed_forward_block,
        dropout
    ):
        super().__init__()

        self.self_attention_block = (
            self_attention_block
        )

        self.cross_attention_block = (
            cross_attention_block
        )

        self.feed_forward_block = (
            feed_forward_block
        )

        self.residual_connections = nn.ModuleList(

            [
                ResidualConnection(dropout),

                ResidualConnection(dropout),

                ResidualConnection(dropout)
            ]

        )

    def forward(
                self,
                x,
                encoder_output,
                src_mask,
                tgt_mask
    ):


# Masked Self Attention

            x = self.residual_connections[0](
                x,
                lambda x:

                self.self_attention_block(
                    x,
                    tgt_mask

                )[0]

            )


# Cross Attention
            x = self.residual_connections[1](

                x,

                lambda x:

                self.cross_attention_block(

                    x,

                    encoder_output,

                    encoder_output,

                    src_mask

                )[0]

            )

# FeedForward

            x = self.residual_connections[2](

                x,

                self.feed_forward_block

            )

            return x


# ==================================================
# Decoder
# ==================================================

class Decoder(nn.Module):

    def __init__(
        self,
        layers
     ):
        super().__init__()

        self.layers = layers

        self.norm = LayerNormalization()


    # Forward Function

    def forward(

            self,

            x,

            encoder_output,

            src_mask,

            tgt_mask

    ):




    # Pass Through All Decoder Blocks

        for i, layer in enumerate(self.layers):

                x = layer(

                x,

                encoder_output,

                src_mask,

                tgt_mask

            )

                if DEBUG:
                    print(f"\nAfter Decoder Layer{i+1}")
                    print(x.shape)


        # Final LayerNorm

        return self.norm(x)


# =============================
# Projection Class
# ============================

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):

        super().__init__()

        self.proj = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        return self.proj(x)
