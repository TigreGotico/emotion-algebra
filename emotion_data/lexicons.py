from os.path import dirname, join


def get_color(word):
    if word in LEXICON:
        return LEXICON[word]["color"]
    return None


def get_word_emotion(word):
    if word in LEXICON:
        return LEXICON[word]["emotion"]
    return None


def get_sentiment(word):
    if word in LEXICON:
        return LEXICON[word]["sentiment"]
    return None


def get_subjectivity(word):
    if word in LEXICON:
        return LEXICON[word]["subjectivity"]
    return None


def get_orientation(word):
    if word in LEXICON:
        return LEXICON[word]["orientation"]
    return None


def load_lexicon():
    bucket = {}

    lexicon_path = join(dirname(__file__), "word_emotion_lexicon.csv")
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for l in lines[1:]:
            l = l.replace("\n", "")
            word, emotion, color, orientation, sentiment, subjectivity, source = l.split(",")
            bucket[word] = {"emotion": emotion,
                            "color": color,
                            "orientation": orientation,
                            "sentiment": sentiment,
                            "subjectivity": subjectivity,
                            "source": source}
    return bucket


LEXICON = load_lexicon()


if __name__ == "__main__":
    from pprint import pprint

    pprint(LEXICON)