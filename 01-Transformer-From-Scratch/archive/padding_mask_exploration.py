import torch


def create_padding_mask(sequence):

    """
    sequence:
    (batch_size, sequence_length)
    """

    mask = (sequence != 0)

    mask = mask.int()

    return mask



# TEST BLOCK

if __name__ == "__main__":

    batch = torch.tensor(

        [

            [1,4,5,6,2],

            [1,10,11,2,0],

            [1,15,2,0,0]

        ]

    )

    print("Batch")

    print(batch)

    print()

    mask = create_padding_mask(batch)

    print("Padding Mask")

    print(mask)