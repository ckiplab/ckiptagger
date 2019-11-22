import os
import sys

# Suppress as many warnings as possible
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def main():
    # Download data
    data_utils.download_data("./")
    
    # Load model without GPU
    ws = WS("./data")
    pos = POS("./data")
    ner = NER("./data")
    
    # Load model with GPU
    # ws = WS("./data", disable_cuda=False)
    # pos = POS("./data", disable_cuda=False)
    # ner = NER("./data", disable_cuda=False)
    
    # Create custom dictionary
    word_to_weight = {
        "土地公": 1,
        "土地婆": 1,
        "公有": 2,
        "": 1,
        "來亂的": "啦",
        "緯來體育台": 1,
    }
    dictionary = construct_dictionary(word_to_weight)
    print(dictionary)
    
    # Run WS-POS-NER pipeline
    sentence_list = [
        "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
        "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
        "",
        "土地公有政策?？還是土地婆有政策。.",
        "… 你確定嗎… 不要再騙了……",
        "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
        "科長說:1,坪數對人數為1:3。2,可以再增加。",
    ]
    word_sentence_list = ws(sentence_list)
    # word_sentence_list = ws(sentence_list, sentence_segmentation=True)
    # word_sentence_list = ws(sentence_list, recommend_dictionary=dictionary)
    # word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    
    # Release model
    del ws
    del pos
    del ner
    
    # Show results
    def print_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return
    
    for i, sentence in enumerate(sentence_list):
        print()
        print(f"'{sentence}'")
        print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
        for entity in sorted(entity_sentence_list[i]):
            print(entity)
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    
