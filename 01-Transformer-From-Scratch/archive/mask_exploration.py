import torch

# Step 1 - Padding Mask Function

def create_padding_mask(sequence, pad_token=0):

    mask = (sequence != pad_token)

    return mask.int()




# TEST BLOCK

if __name__ == "__main__":

    sequence = torch.tensor(
        [1, 15, 2, 0, 0]
    )

    print("Sequence:")
    print(sequence)
    print()

    mask = create_padding_mask(sequence)

    print("Padding Mask:")
    print(mask)
