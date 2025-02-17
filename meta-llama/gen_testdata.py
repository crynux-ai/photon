from tokenizer import Tokenizer
import model

import torch
import struct

def encode(content: str) -> str:
    t = Tokenizer("models/tokenizer.model")
    res = t.encode(content, True, True)
    return ", ".join(map(str, res))


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

    token = [encode(t) for t in text]

    with open("unit_tests/testdata/sentencepiece.dat", "w") as file:
        for i in range(len(text)):
            file.write(text[i])
            file.write("\n")
            file.write(token[i])
            file.write("\n")

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
    x = torch.randn(1, 256)
    y = layer(x)
    print(y.shape)
    with open("unit_tests/testdata/feedforward.dat", "wb") as file:
        file.write(struct.pack("i", 256) + struct.pack("i", 1024) + struct.pack("i", 4))
        file.write(tensor_to_bytes(layer.w1.weight))
        file.write(tensor_to_bytes(layer.w2.weight))
        file.write(tensor_to_bytes(layer.w3.weight))
        file.write(tensor_to_bytes(x))
        file.write(tensor_to_bytes(y))


if __name__ == '__main__':
    gen_tokenizer_data()
    gen_feedforward_data()