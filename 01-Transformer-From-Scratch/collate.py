import torch

from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    encoder_inputs = []

    decoder_inputs = []

    labels = []

    # --------------------------
    # Separate every sample
    # --------------------------

    for encoder_input, decoder_input, label in batch:

        encoder_inputs.append(encoder_input)

        decoder_inputs.append(decoder_input)

        labels.append(label)

    # --------------------------
    # Pad Encoder
    # --------------------------

    encoder_inputs = pad_sequence(

        encoder_inputs,

        batch_first=True,

        padding_value=0

    )

    # --------------------------
    # Pad Decoder
    # --------------------------

    decoder_inputs = pad_sequence(

        decoder_inputs,

        batch_first=True,

        padding_value=0

    )

    # --------------------------
    # Pad Labels
    # --------------------------

    labels = pad_sequence(

        labels,

        batch_first=True,

        padding_value=0

    )

    return (

        encoder_inputs,

        decoder_inputs,

        labels

    )