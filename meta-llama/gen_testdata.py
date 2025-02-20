from tokenizer import Tokenizer
import model

import torch
import time
import struct

def tokenize(content: str) -> list[int]:
    tokenizer = Tokenizer("models/tokenizer.model")
    res = tokenizer.encode(content, True, True)
    return res


def gen_tokenizer_data():
    text = [
    "",
    " ",
    """A photon (from Ancient Greek φῶς, φωτός (phôs, phōtós) 'light') is an
 elementary particle that is a quantum of the electromagnetic field, including electromagnetic radiation
 such as light and radio waves, and the force carrier for the electromagnetic force. Photons are massless
 particles that can move no faster than the speed of light measured in vacuum. The photon belongs to the
 class of boson particles. As with other elementary particles, photons are best explained by quantum
 mechanics and exhibit wave–particle duality, their behavior featuring properties of both waves and particles.""",
    ]

    token = [tokenize(t) for t in text]

    with open("unit_tests/testdata/sentencepiece.dat", "wb") as file:
        file.write(struct.pack("i", 3))
        for i in range(len(text)):
            btext = text[i].encode("utf-8")
            file.write(struct.pack("i", len(btext)))
            file.write(btext)
            file.write(struct.pack("i", len(token[i])))
            for j in token[i]:
                file.write(struct.pack("i", j))

def tensor_to_bytes(tensor):
    data = struct.pack("i", len(tensor.shape))
    content = [tensor.tolist()]
    for x in tensor.shape:
        data += struct.pack("i", x)
        content = sum(content, [])
    for x in content:
        data += struct.pack("f", x)
    return data


def gen_feedforward_data():
    layer = model.FeedForward(dim=256, hidden_dim=1024, multiple_of=4, ffn_dim_multiplier=None)
    x = torch.randn(3, 7, 256)
    y = layer(x)
    with open("unit_tests/testdata/feedforward.dat", "wb") as file:
        file.write(struct.pack("i", 256) + struct.pack("i", 1024) + struct.pack("i", 4))
        file.write(tensor_to_bytes(layer.w1.weight))
        file.write(tensor_to_bytes(layer.w2.weight))
        file.write(tensor_to_bytes(layer.w3.weight))
        file.write(tensor_to_bytes(x))
        file.write(tensor_to_bytes(y))

def mask(seqlen, start_pos):
    mask = torch.full((seqlen, seqlen), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    mask = torch.hstack([
        torch.zeros((seqlen, start_pos)),
        mask
    ])
    return mask


def gen_attention_data():
    freqs_cis = model.precompute_freqs_cis(64, 99, 10000.0)
    layer = model.Attention(model.ModelArgs(dim=256, n_heads=4, multiple_of=4))

    x1 = torch.randn(3, 7, 256)
    y1 = layer.forward(x1, 0, freqs_cis[0: 7], mask(7, 0))

    x2 = torch.randn(3, 3, 256)
    y2 = layer.forward(x2, 7, freqs_cis[7: 10], mask(3, 7))

    x3 = torch.randn(3, 2, 256)
    y3 = layer.forward(x3, 10, freqs_cis[10: 12], None)

    with open("unit_tests/testdata/attention.dat", "wb") as file:
        file.write(struct.pack("i", 256) + struct.pack("i", 4) + struct.pack("i", 4))  # layer
        file.write(struct.pack("i", 64) + struct.pack("i", 99) + struct.pack("f", 10000.0)) # freqs
        file.write(tensor_to_bytes(layer.wq.weight))
        file.write(tensor_to_bytes(layer.wk.weight))
        file.write(tensor_to_bytes(layer.wv.weight))
        file.write(tensor_to_bytes(layer.wo.weight))
        file.write(tensor_to_bytes(x1))
        file.write(tensor_to_bytes(y1))
        file.write(tensor_to_bytes(x2))
        file.write(tensor_to_bytes(y2))
        file.write(tensor_to_bytes(x3))
        file.write(tensor_to_bytes(y3))


def gen_rope_data():
    freqs_cis = model.precompute_freqs_cis(32, 77, 10000.0)
    xq = torch.randn(3, 7, 4, 32)
    xk = torch.randn(3, 7, 4, 32)
    pq, pk = model.apply_rotary_emb(xq, xk, freqs_cis[10:17])

    with open("unit_tests/testdata/rope.dat", "wb") as file:
        file.write(struct.pack("i", 3) + struct.pack("i", 77))
        file.write(struct.pack("i", 10) + struct.pack("i", 7))
        file.write(struct.pack("i", 4) + struct.pack("i", 32))
        file.write(tensor_to_bytes(xq.view(3, 7, 128)))
        file.write(tensor_to_bytes(xk.view(3, 7, 128)))
        file.write(tensor_to_bytes(pq.view(3, 7, 128)))
        file.write(tensor_to_bytes(pk.view(3, 7, 128)))


def gen_transformer_data():
    freqs_cis = model.precompute_freqs_cis(32, 77, 10000.0)
    layer = model.Transformer(model.ModelArgs(
        dim=128,
        n_layers=3,
        n_heads=4,
        multiple_of=4,
        vocab_size=1037))

    with open("unit_tests/testdata/transformer.dat", "wb") as file:
        file.write(struct.pack("i", 32) + struct.pack("i", 77))
        file.write(struct.pack("i", 128) + struct.pack("i", 3) + struct.pack("i", 4))
        file.write(struct.pack("i", 4) + struct.pack("i", 1037))
        file.write(struct.pack("i", 3))
        file.write(struct.pack("i", 3) + struct.pack("i", 7))
        file.write(struct.pack("i", 3) + struct.pack("i", 3))
        file.write(struct.pack("i", 3) + struct.pack("i", 2))
        for i in range(3):
            attention = layer.layers[i].attention
            ffn = layer.layers[i].feed_forward
            file.write(tensor_to_bytes(attention.wq.weight))
            file.write(tensor_to_bytes(attention.wk.weight))
            file.write(tensor_to_bytes(attention.wv.weight))
            file.write(tensor_to_bytes(attention.wo.weight))
            file.write(tensor_to_bytes(ffn.w1.weight))
            file.write(tensor_to_bytes(ffn.w2.weight))
            file.write(tensor_to_bytes(ffn.w3.weight))
        file.write(tensor_to_bytes(layer.tok_embeddings.weight))
        file.write(tensor_to_bytes(layer.output.weight))

        token = torch.randint(0, 1037, (3, 7))
        start = time.time()
        output = layer.forward(token, 0)
        print("time: ", time.time() - start)
        for i in range(3):
            for j in range(7):
                file.write(struct.pack("i", token[i][j].item()))
        file.write(tensor_to_bytes(output))

        token = torch.randint(0, 1037, (3, 3))
        output = layer.forward(token, 7)
        for i in range(3):
            for j in range(3):
                file.write(struct.pack("i", token[i][j].item()))
        file.write(tensor_to_bytes(output))

        token = torch.randint(0, 1037, (3, 2))
        output = layer.forward(token, 10)
        for i in range(3):
            for j in range(2):
                file.write(struct.pack("i", token[i][j].item()))
        file.write(tensor_to_bytes(output))


if __name__ == '__main__':
    # gen_tokenizer_data()
    # gen_feedforward_data()
    # gen_rope_data()
    # gen_attention_data()
    gen_transformer_data()