from tokenizer import Tokenizer


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


if __name__ == '__main__':
    gen_tokenizer_data()