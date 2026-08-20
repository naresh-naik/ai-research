import torch
import torch.nn as nn

from model import (
    InputEmbeddings,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForwardBlock,
    EncoderBlock,
    Encoder,
)
from decoder import (
    MaskedMultiHeadAttention,
    CrossMultiHeadAttention,
    DecoderBlock,
    Decoder,
    ProjectionLayer,
)


# Step 1 - Transformer Module

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        d_model,
        N,
        h,
        dropout,
        d_ff,
    ):
        super().__init__()

        self.src_embed = nn.Sequential(
            InputEmbeddings(d_model, src_vocab_size),
            PositionalEncoding(d_model, src_seq_len, dropout),
        )

        self.tgt_embed = nn.Sequential(
            InputEmbeddings(d_model, tgt_vocab_size),
            PositionalEncoding(d_model, tgt_seq_len, dropout),
        )

        self.encoder = Encoder(
            nn.ModuleList(
                [
                    EncoderBlock(
                        MultiHeadAttention(d_model, h),
                        FeedForwardBlock(d_model, d_ff, dropout),
                        dropout,
                    )
                    for _ in range(N)
                ]
            )
        )

        self.decoder = Decoder(
            nn.ModuleList(
                [
                    DecoderBlock(
                        MaskedMultiHeadAttention(d_model, h),
                        CrossMultiHeadAttention(d_model, h),
                        FeedForwardBlock(d_model, d_ff, dropout),
                        dropout,
                    )
                    for _ in range(N)
                ]
            )
        )

        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):

        return self.encoder(self.src_embed(src))

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):

        return self.decoder(
            self.tgt_embed(tgt),
            encoder_output,
            src_mask,
            tgt_mask,
        )

    def project(self, x):

        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        if tgt_mask is None:
            tgt_mask = torch.tril(
                torch.ones(
                    tgt.shape[1],
                    tgt.shape[1],
                    device=tgt.device,
                )
            )

        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(
            encoder_output,
            src_mask,
            tgt,
            tgt_mask,
        )

        return self.project(decoder_output)