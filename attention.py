import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def create_padding_mask(mask):
    return mask[:, np.newaxis, :]

def scaled_dot_product_attention(Q, K, V, mask=None):

    scores = np.matmul(Q, K.transpose(0, 2, 1))
    dk = Q.shape[-1]
    scaled_scores = scores / np.sqrt(dk)

    if mask is not None:
        scaled_scores = scaled_scores + (mask * -1e9)

    attention_weights = softmax(scaled_scores)
    output = np.matmul(attention_weights, V)

    return output, attention_weights

if __name__ == "__main__":

    sentence = input("Enter a sentence: ")
    words = sentence.split()

    batch_size = 1
    seq_len = len(words)
    embed_dim = 8

    embeddings = np.random.randn(batch_size, seq_len, embed_dim)

    Q = embeddings
    K = embeddings
    V = embeddings
    
    padding_mask = np.ones((1, seq_len))
    mask = create_padding_mask(1 - padding_mask)

    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

    print("\nOutput Shape:", output.shape)
    print("\nOutput Tensor:\n", output)

    print("\nAttention Per Word")
    attn = attention_weights[0]

    for i, word in enumerate(words):
        most_attended_index = np.argmax(attn[i])
        print(f"{word} attends most to -> {words[most_attended_index]} "
              f"(score={attn[i][most_attended_index]:.3f})")
        
    plt.imshow(attention_weights[0])
    plt.xticks(range(seq_len), words, rotation=45)
    plt.yticks(range(seq_len), words)
    plt.colorbar()
    plt.title("Attention Heatmap")
    plt.show()
