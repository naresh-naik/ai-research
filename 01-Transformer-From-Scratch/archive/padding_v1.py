import torch


# Step 1 - Padding Function

def pad_sequences(sequences, pad_value=0):

	# Find the longest sequence
	max_length = max(len(seq) for seq in sequences)

	padded_sequences = []

	for seq in sequences:

		padding = [pad_value] * (max_length - len(seq))

		padded_seq = torch.cat(
			[
				seq,
				torch.tensor(
					padding,
					dtype=torch.long
				)
			]
		)

		padded_sequences.append(padded_seq)

	return torch.stack(padded_sequences)


if __name__ == "__main__":

	seq1 = torch.tensor([1, 4, 5, 6, 2])
	seq2 = torch.tensor([1, 10, 11, 2])
	seq3 = torch.tensor([1, 15, 2])

	padded = pad_sequences(
		[
			seq1,
			seq2,
			seq3
		]
	)

	print("Padded Batch:")
	print(padded)
	print()
	print("Shape:")
	print(padded.shape)


