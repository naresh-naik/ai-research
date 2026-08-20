import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from tokenizer import Tokenizer
from dataset import TranslationDataset, training_pairs
from padding import pad_sequences
from transformer import Transformer


# Step 1 - Build Vocabulary

vocab = Vocabulary()

vocab.add_special_tokens()

all_sentences = []

for src, tgt in training_pairs:
	all_sentences.append(src)
	all_sentences.append(tgt)

vocab.build_vocabulary(all_sentences)


# Step 2 - Tokenizer

tokenizer = Tokenizer(vocab)


# Step 3 - Dataset

dataset = TranslationDataset(
	training_pairs,
	tokenizer,
)


# Step 4 - DataLoader


# Step 4.1 - Collate Function

def collate_batch(batch):

	encoder_inputs = pad_sequences(
		[item["encoder_input"] for item in batch]
	)

	decoder_inputs = pad_sequences(
		[item["decoder_input"] for item in batch]
	)

	labels = pad_sequences(
		[item["label"] for item in batch]
	)

	return {
		"encoder_input": encoder_inputs,
		"decoder_input": decoder_inputs,
		"label": labels,
	}

loader = DataLoader(
	dataset,
	batch_size=2,
	shuffle=True,
	collate_fn=collate_batch,
)


# Step 5 - Build Transformer

transformer = Transformer(
	src_vocab_size=len(vocab.word_to_id),
	tgt_vocab_size=len(vocab.word_to_id),
	src_seq_len=20,
	tgt_seq_len=20,
	d_model=512,
	N=6,
	h=8,
	dropout=0.1,
	d_ff=2048,
)


# Step 6 - Loss Function

criterion = nn.CrossEntropyLoss(
	ignore_index=0,
)



# Step 7 - Optimizer

optimizer = torch.optim.Adam(
	transformer.parameters(),
	lr=0.0001,
)



if __name__ == "__main__":

	print("Training scaffold ready.")
	print(f"Vocabulary size: {len(vocab.word_to_id)}")
	print(f"Dataset size: {len(dataset)}")
	print(f"Batch size: {loader.batch_size}")

	# Step 9 - Training Loop

	num_epochs = 10

	for epoch in range(num_epochs):

		transformer.train()

		total_loss = 0

		for batch in loader:

			encoder_input = batch["encoder_input"]
			decoder_input = batch["decoder_input"]
			labels = batch["label"]

			print("\n==============================")
			print("NEW BATCH")
			print("==============================")

			print("Encoder Input Shape:")
			print(encoder_input.shape)

			print("Decoder Input Shape:")
			print(decoder_input.shape)

			print("Labels Shape:")
			print(labels.shape)

			logits = transformer(
				encoder_input,
				decoder_input,
			)

			print("\nLogits Shape:")
			print(logits.shape)

			logits = logits.view(
				-1,
				logits.shape[-1]
			)

			labels = labels.view(-1)

			loss = criterion(
				logits,
				labels,
			)

			print()
			print("Flattened Logits Shape:")
			print(logits.shape)
			print()
			print("Flattened Labels Shape:")
			print(labels.shape)
			print()
			print("Loss:")
			print(loss.item())

			optimizer.zero_grad()

			loss.backward()

			print()
			print("Gradient Shape:")
			print(
				transformer.encoder.layers[0]
				.self_attention_block
				.w_q.weight.grad.shape
			)

			optimizer.step()

			print()
			print("Weights Updated Successfully!")
			break

		break




