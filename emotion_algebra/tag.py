import requests


# DO NOT abuse this, meant for dev purposes, you should use the official api not hijack the demo site

def tag(text, lang="en-us"):
    try:
        data = {"lang_code": lang, "text": text, "api_type": "emotion"}
        data = requests.post("https://www.paralleldots.com/api/demos", data).json()["emotion"]["probabilities"]
        result = {}
        for e in data:
            if e.lower() == "happy":
                fixed = "joy"
            elif e.lower() == "excited":
                fixed = "anticipation"
            elif e.lower() == "sad":
                fixed = "sadness"
            elif e.lower() == "angry":
                fixed = "anger"
            elif e.lower() == "bored":
                fixed = "boredom"
            elif e.lower() == "sarcasm":
                fixed = "annoyance"
            result[fixed] = data[e]
        return result
    except:
        return {}


def best_emotion(text, lang="en-us"):
    data = tag(text, lang)
    top_score = 0
    best = ""
    for e in data:
        score = data[e]
        if score > top_score:
            best = e
            top_score = score
    return best


def test():
    TEST_SENTENCES = ['I love mom\'s cooking',  # happy
                      'I love how you never reply back..',  # sarcasm
                      'I love cruising with my homies',  # excited
                      'I love messing with yo mind!!',  # fear
                      'I love you and now you\'re just gone..',  # sad
                      'This is shit',  # angry
                      'This is the shit']  # excited
    for t in TEST_SENTENCES:
        print(best_emotion(t))

