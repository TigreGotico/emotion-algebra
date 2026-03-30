import random
from emotion_data.plutchik import DIMENSIONS


def _get_emotion_map():
    bucket = {}

    # get the basic emotions from each dimension
    for dimension_name in DIMENSIONS:
        dimension = DIMENSIONS[dimension_name]
        bucket[dimension.basic_emotion.name] = dimension.basic_emotion
        bucket[dimension.basic_opposite.name] = dimension.basic_opposite
        bucket[dimension.mild_emotion.name] = dimension.mild_emotion
        bucket[dimension.mild_opposite.name] = dimension.mild_opposite
        bucket[dimension.intense_emotion.name] = dimension.intense_emotion
        bucket[dimension.intense_opposite.name] = dimension.intense_opposite
    return bucket


EMOTIONS = _get_emotion_map()

EMOTION_NAMES = [EMOTIONS[e].name for e in EMOTIONS]

POSITIVE_EMOTIONS = [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].valence > 0]

NEGATIVE_EMOTIONS = [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].valence < 0]

DIMENSION_TO_EMOTION_MAP = {
    "sensitivity": [EMOTIONS[e] for e in EMOTIONS if
                    EMOTIONS[e].dimension and EMOTIONS[e].dimension.name == "sensitivity"],
    "attention": [EMOTIONS[e] for e in EMOTIONS if
                  EMOTIONS[e].dimension and EMOTIONS[e].dimension.name == "attention"],
    "pleasantness": [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].dimension and
                     EMOTIONS[e].dimension.name == "pleasantness"],
    "aptitude": [EMOTIONS[e] for e in EMOTIONS if
                 EMOTIONS[e].dimension and EMOTIONS[e].dimension.name == "aptitude"]
}

KIND_TO_EMOTION_MAP = {
    "related to object properties":
        [EMOTIONS[e] for e in EMOTIONS if
         EMOTIONS[e].kind == "related to object properties"],
    'future appraisal': [EMOTIONS[e] for e in EMOTIONS if
                         EMOTIONS[e].kind == "future appraisal"],
    'event related': [EMOTIONS[e] for e in EMOTIONS if
                      EMOTIONS[e].kind == "event related"],
    'self appraisal': [EMOTIONS[e] for e in EMOTIONS if
                       EMOTIONS[e].kind == "self appraisal"],
    'social': [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].kind == "social"],
    'cathected': [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].kind == "cathected"]
}

# TODO emotion type map


def random_emotion():
    return EMOTIONS.get(random.choice(list(EMOTIONS.keys())))


def get_emotion(emotion_name):
    return EMOTIONS.get(emotion_name)


def get_dimension(dimension_name):
    return DIMENSIONS.get(dimension_name)


def emotion_to_dimension(emotion_name):
    emotion = get_emotion(emotion_name)
    if emotion is not None:
        if emotion.is_composite:
            return [e.dimension for e in emotion.components if e.dimension]
        return emotion.dimension
    return None


if __name__ == "__main__":
    from pprint import pprint

    print(EMOTION_NAMES)
    print(POSITIVE_EMOTIONS)
    print(NEGATIVE_EMOTIONS)
    pprint(EMOTIONS)
    pprint(KIND_TO_EMOTION_MAP)
    pprint(DIMENSION_TO_EMOTION_MAP)
