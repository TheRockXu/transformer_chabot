import tensorflow as tf
from train import *
import matplotlib.pyplot as plt
MAX_LENGTH=150

def evaluate(inp_sentence):
    train_dataset, val_dataset, tokenizer = build_datasets()
    vocab_size = tokenizer.vocab_size + 2
    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              vocab_size, vocab_size,
                              pe_input=vocab_size,
                              pe_target=vocab_size,
                              rate=dropout_rate)
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=tf.keras.optimizers.SGD())

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')


    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    train_dataset, val_dataset, tokenizer = build_datasets()
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer.decode([i]) for i in result
                            if i < tokenizer.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def answer(question, plot=''):
    train_dataset, val_dataset, tokenizer = build_datasets()
    result, attention_weights = evaluate(question)

    predicted_sentence = tokenizer.decode([i for i in result
                                              if i < tokenizer.vocab_size])

    print('Input: {}'.format(question))
    print('Predicted answer: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, question, result, plot)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Answer Financial Questions')
    parser.add_argument('question',  type=str,
                        help='ask a question')
    # Test model
    args = parser.parse_args()

    answer(args.question)