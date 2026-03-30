# deepmoji

tag a sentence with emojis

NOTE: DO NOT abuse this, meant for dev purposes, you should deploy your own deepmoji not hijack the [demo site](https://deepmoji.mit.edu/)!

        from deepmoji import EMOJI_MAP, get_emojis, get_emoji_scores
        
        TEST_SENTENCES = ['I love mom\'s cooking',
                              'I love how you never reply back..',
                              'I love cruising with my homies',
                              'I love messing with yo mind!!',
                              'I love you and now you\'re just gone..',
                              'This is shit',
                              'This is the shit']
        for t in TEST_SENTENCES:
            print(get_emojis(t))
            
        """
        [':stuck_out_tongue_closed_eyes:', ':heart_eyes:', ':heart:', ':blush:', ':yellow_heart:']
        [':unamused:', ':expressionless:', ':angry:', ':neutral_face:', ':broken_heart:']
        [':sunglasses:', ':ok_hand:', ':v:', ':relieved:', ':100:']
        [':stuck_out_tongue_winking_eye:', ':smiling_imp:', ':smirk:', ':wink:', ':speak_no_evil:']
        [':broken_heart:', ':pensive:', ':disappointed:', ':sleepy:', ':cry:']
        [':angry:', ':rage:', ':disappointed:', ':unamused:', ':triumph:']
        [':headphones:', ':notes:', ':ok_hand:', ':sunglasses:', ':smirk:']
        """
        
        print(get_emoji_scores(TEST_SENTENCES[0]))
        
        """
        {8: 0.031, 16: 0.03, 47: 0.029, 4: 0.088, 36: 0.491}
        """
        
        emoji = EMOJI_MAP[4]
        print(emoji)  # :heart_eyes:
        
         
# parallel dots

assigning a emotion to a sentence can be complicated, for development purposes paralleldots is being used

NOTE: DO NOT abuse this, meant for dev purposes, you should [get a key](https://www.paralleldots.com/api-wrappers) not hijack the [demo site](https://www.paralleldots.com/text-analysis-apis)!

     from emotion_data.tag import tag
     
     TEST_SENTENCES = ['I love mom\'s cooking', #happy
                      'I love how you never reply back..', # sarcasm
                      'I love cruising with my homies', # excited
                      'I love messing with yo mind!!', # fear
                      'I love you and now you\'re just gone..', #sad
                      'This is shit', # angry
                      'This is the shit'] # excited
     for t in TEST_SENTENCES:
        print(tag(t))
        
     """
     {'Excited': 0.21985779096584576, 'Angry': 0.007277394240280612, 'Fear': 0.028446538283358705, 'Happy': 0.6123207400377838, 'Sad': 0.05338429360950592, 'Bored': 0.003940096398789999, 'Sarcasm': 0.07477314646443523}
     {'Excited': 0.09110381778948567, 'Angry': 0.2232724453005621, 'Fear': 0.08151735485701023, 'Happy': 0.046755292242729944, 'Sad': 0.20477872670117295, 'Bored': 0.10956273286363101, 'Sarcasm': 0.24300963024540798}
     {'Excited': 0.6214024076233992, 'Angry': 0.03832698424974395, 'Fear': 0.09534910180969712, 'Happy': 0.16127235757202596, 'Sad': 0.004707877457760142, 'Bored': 0.00706969981103465, 'Sarcasm': 0.07187157147633895}
     {'Excited': 0.11859858286096328, 'Angry': 0.04123953737969006, 'Fear': 0.31777673118239985, 'Happy': 0.24683752452264218, 'Sad': 0.028046879736793935, 'Bored': 0.01289525656438939, 'Sarcasm': 0.23460548775312118}
     {'Excited': 0.019495569949675072, 'Angry': 0.03652994103485607, 'Fear': 0.04060912511109711, 'Happy': 0.0571281335897157, 'Sad': 0.7298586811819636, 'Bored': 0.05963816472466741, 'Sarcasm': 0.05674038440802512}
     {'Excited': 0.02622683078142861, 'Angry': 0.2935166785519792, 'Fear': 0.10385564059215835, 'Happy': 0.01670497783731631, 'Sad': 0.26638222751025986, 'Bored': 0.1718336543018634, 'Sarcasm': 0.12147999042499436}
     {'Excited': 0.33323072746382143, 'Angry': 0.08253854853741412, 'Fear': 0.15102825676434467, 'Happy': 0.20941643042277513, 'Sad': 0.0308394455870149, 'Bored': 0.022478864823863225, 'Sarcasm': 0.17046772640076668}
     """
        
# future

emojis will be associated with feelings, and used to attach emotional information to sentences