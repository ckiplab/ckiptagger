import re
import os
import sys
import unicodedata

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ckiptagger import model_ws
from ckiptagger import model_pos
from ckiptagger import model_ner

_whitespace_pattern = re.compile("[\s]+")

def construct_dictionary(word_to_weight):
    length_word_weight = {}
    
    for word, weight in word_to_weight.items():
        if not word: continue
        try:
            weight = float(weight)
        except ValueError:
            continue
        length = len(word)
        if length not in length_word_weight:
            length_word_weight[length] = {}
        length_word_weight[length][word] = weight
        
    length_word_weight = sorted(length_word_weight.items())
    
    return length_word_weight
    
class WS:
    def __init__(self, data_dir, disable_cuda=True):
        config = model_ws.Config()
        config.name = "model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1"
        config.attention_heads = 0
        config.is_cross_bilstm = True
        config.layers = 2
        config.hidden_d = 500
        config.w_token_to_vector, config.w_embedding_d = _load_embedding(os.path.join(data_dir, "embedding_character"))
        
        if disable_cuda:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                env_backup = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                env_backup = None
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
        with tf.Graph().as_default():
            model = model_ws.Model(config)
            model.sess = tf.compat.v1.Session()
            model.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(model.sess, os.path.join(data_dir, "model_ws", config.name))
            
        if disable_cuda and env_backup:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
            
        self.model = model
        return
        
    def __del__(self):
        self.model.sess.close()
        return
        
    def __call__(
            self,
            sentence_list,
            
            recommend_dictionary = {},
            coerce_dictionary = {},
            
            sentence_segmentation = False,
            segment_delimiter_set = {",", "。", ":", "?", "!", ";"},
            
            character_normalization = True,
            batch_sentences = 2048,
            batch_characters = 16384,
        ):
        
        # Character normalization
        if character_normalization:
            raw_sentence_list = sentence_list
            sentence_list = []
            normal2raw_list = []
            for raw_sentence in raw_sentence_list:
                sentence, normal_to_raw_index = _normalize_sentence(raw_sentence)
                sentence_list.append(sentence)
                normal2raw_list.append(normal_to_raw_index)
            segment_delimiter_set = {unicodedata.normalize("NFKD", word) for word in segment_delimiter_set}
            
        # Sentence segmentation
        segment_list = []
        segment_to_sentence_index = []
        for sentence_index, sentence in enumerate(sentence_list):
            if sentence_segmentation:
                partial_segment_list = _segment_sentence(sentence, segment_delimiter_set)
            else:
                partial_segment_list = [sentence]
            for segment in partial_segment_list:
                segment_list.append(segment)
                segment_to_sentence_index.append(sentence_index)
                
        # Sequence-label prediction
        batch_list = _get_ws_batch_list(segment_list, batch_sentences, batch_characters)
        seq_segment_list = [None] * len(segment_list)
        for batch in batch_list:
            index_list, sample_list = zip(*batch)
            if len(sample_list[0][0]) == 0:
                parital_seq_segment_list = [[] for sample in sample_list]
            else:
                parital_seq_segment_list = self.model.predict_label_for_a_batch(sample_list)
            for partial_index, index in enumerate(index_list):
                seq_segment_list[index] = parital_seq_segment_list[partial_index]
                
        # Undo sentence segmentation
        seq_sentence_list = []
        seq_sentence = []
        sentence_index = 0
        for segment_index, seq_segment in enumerate(seq_segment_list):
            assert len(segment_list[segment_index]) == len(seq_segment)
            if sentence_index != segment_to_sentence_index[segment_index]:
                seq_sentence_list.append(seq_sentence)
                seq_sentence = []
                sentence_index = segment_to_sentence_index[segment_index]
            seq_sentence += seq_segment
        seq_sentence_list.append(seq_sentence)
        
        # Undo character normalization
        if character_normalization:
            sentence_list = raw_sentence_list
            raw_seq_sentence_list = []
            for seq_sentence, normal_to_raw_index in zip(seq_sentence_list, normal2raw_list):
                raw_seq_sentence = []
                for index, label in enumerate(seq_sentence):
                    if normal_to_raw_index[index] < 0: continue
                    raw_seq_sentence.append(label)
                raw_seq_sentence_list.append(raw_seq_sentence)
            seq_sentence_list = raw_seq_sentence_list
            
        # Word segmentation
        word_sentence_list = []
        for sentence, seq_segment in zip(sentence_list, seq_sentence_list):
            word_sentence = _get_word_sentence_from_seq_sentence(sentence, seq_segment)
            word_sentence = _run_word_segmentation_with_dictionary(word_sentence, recommend_dictionary, coerce_dictionary)
            word_sentence_list.append(word_sentence)
            
        return word_sentence_list
        
class POS:
    def __init__(self, data_dir, disable_cuda=True):
        config = model_pos.Config()
        config.name = "model_asbc_Att-0_BiLSTM-2-500_batch256-run1"
        config.attention_heads = 0
        config.layers = 2
        config.hidden_d = 500
        config.c_token_to_vector, config.c_embedding_d = _load_embedding(os.path.join(data_dir, "embedding_character"))
        config.w_token_to_vector, config.w_embedding_d = _load_embedding(os.path.join(data_dir, "embedding_word"))
        config.label_list, config.label_to_index = _read_pos_list(os.path.join(data_dir, "model_pos", "label_list.txt"))
        config.output_d = len(config.label_list)
        
        if disable_cuda:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                env_backup = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                env_backup = None
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
        with tf.Graph().as_default():
            model = model_pos.Model(config)
            model.sess = tf.compat.v1.Session()
            model.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(model.sess, os.path.join(data_dir, "model_pos", config.name))
            
        if disable_cuda and env_backup:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
            
        self.model = model
        return
        
    def __del__(self):
        self.model.sess.close()
        return
        
    def __call__(
            self,
            sentence_list,
            
            sentence_segmentation = True,
            segment_delimiter_set = {",", "。", ":", "?", "!", ";"},
            
            character_normalization = True,
            batch_sentences = 2048,
            batch_characters = 16384,
        ):
        
        # Character normalization
        if character_normalization:
            raw_sentence_list = sentence_list
            sentence_list = []
            for raw_sentence in raw_sentence_list:
                sentence = []
                for raw_word in raw_sentence:
                    word = unicodedata.normalize("NFKD", raw_word)
                    sentence.append(word)
                sentence_list.append(sentence)
                
        # Sentence segmentation
        segment_list = []
        segment_to_sentence_index = []
        for sentence_index, sentence in enumerate(sentence_list):
            if sentence_segmentation:
                partial_segment_list = _segment_word_sentence(sentence, segment_delimiter_set)
            else:
                partial_segment_list = [sentence]
            for segment in partial_segment_list:
                segment_list.append(segment)
                segment_to_sentence_index.append(sentence_index)
                
        # POS prediction
        batch_list = _get_pos_batch_list(segment_list, batch_sentences, batch_characters)
        pos_segment_list = [None] * len(segment_list)
        for batch in batch_list:
            index_list, sample_list = zip(*batch)
            if len(sample_list[0][0]) == 0:
                parital_pos_segment_list = [[] for sample in sample_list]
            else:
                parital_pos_segment_list = self.model.predict_label_for_a_batch(sample_list)
            for partial_index, index in enumerate(index_list):
                pos_segment_list[index] = parital_pos_segment_list[partial_index]
                
        # Undo sentence segmentation
        pos_sentence_list = []
        pos_sentence = []
        sentence_index = 0
        for segment_index, pos_segment in enumerate(pos_segment_list):
            assert len(segment_list[segment_index]) == len(pos_segment)
            _force_whitespace_tagging(segment_list[segment_index], pos_segment)
            
            if sentence_index != segment_to_sentence_index[segment_index]:
                pos_sentence_list.append(pos_sentence)
                pos_sentence = []
                sentence_index = segment_to_sentence_index[segment_index]
            pos_sentence += pos_segment
        pos_sentence_list.append(pos_sentence)
        
        return pos_sentence_list
        
class NER:
    def __init__(self, data_dir, disable_cuda=True):
        config = model_ner.Config()
        config.name = "model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1"
        config.attention_heads = 0
        config.layers = 2
        config.hidden_d = 500
        config.c_token_to_vector, config.c_embedding_d = _load_embedding(os.path.join(data_dir, "embedding_character"))
        config.w_token_to_vector, config.w_embedding_d = _load_embedding(os.path.join(data_dir, "embedding_word"))
        _, config.pos_to_index = _read_pos_list(os.path.join(data_dir, "model_ner", "pos_list.txt"))
        config.w_feature_d = len(config.pos_to_index)
        config.label_list, config.label_to_index = _read_entity_type_list(os.path.join(data_dir, "model_ner", "label_list.txt"))
        config.output_d = len(config.label_list)
        
        if disable_cuda:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                env_backup = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                env_backup = None
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
        with tf.Graph().as_default():
            model = model_ner.Model(config)
            model.sess = tf.compat.v1.Session()
            model.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(model.sess, os.path.join(data_dir, "model_ner", config.name))
            
        if disable_cuda and env_backup:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
            
        self.model = model
        return
        
    def __del__(self):
        self.model.sess.close()
        return
        
    def __call__(
            self,
            word_sentence_list,
            pos_sentence_list,
            
            character_normalization = True,
            batch_sentences = 2048,
            batch_characters = 16384,
        ):
        
        # Character normalization
        if character_normalization:
            raw_word_sentence_list = word_sentence_list
            word_sentence_list = []
            normal2raw_list = []
            for raw_word_sentence in raw_word_sentence_list:
                word_sentence = [unicodedata.normalize("NFKD", raw_word) for raw_word in raw_word_sentence]
                sentence, normal_to_raw_index = _normalize_sentence("".join(raw_word_sentence))
                assert sentence == "".join(word_sentence)
                word_sentence_list.append(word_sentence)
                normal2raw_list.append(normal_to_raw_index)
                
        # Sequence-label prediction
        batch_list = _get_ner_batch_list(word_sentence_list, pos_sentence_list, batch_sentences, batch_characters)
        label_sentence_list = [None] * len(word_sentence_list)
        for batch in batch_list:
            index_list, sample_list = zip(*batch)
            if len(sample_list[0][0]) == 0:
                parital_label_sentence_list = [[] for sample in sample_list]
            else:
                parital_label_sentence_list = self.model.predict_label_for_a_batch(sample_list)
            for partial_index, index in enumerate(index_list):
                label_sentence_list[index] = parital_label_sentence_list[partial_index]
                
        # Undo character normalization
        if character_normalization:
            word_sentence_list = raw_word_sentence_list
            raw_label_sentence_list = []
            for label_sentence, normal_to_raw_index in zip(label_sentence_list, normal2raw_list):
                raw_label_sentence = []
                for index, label in enumerate(label_sentence):
                    if normal_to_raw_index[index] < 0:
                        if label[-1]!="E" or raw_label_sentence[-1]=="O": continue
                        entity_type, seq_type = raw_label_sentence[-1].split(":")
                        if entity_type != label[:-2]: continue
                        if seq_type == "B":
                            raw_label_sentence[-1] = entity_type + ":S"
                        elif seq_type == "I":
                            raw_label_sentence[-1] = entity_type + ":E"
                        continue
                    raw_label_sentence.append(label)
                raw_label_sentence_list.append(raw_label_sentence)
            label_sentence_list = raw_label_sentence_list
        
        # Entity mention collection
        entity_sentence_list = []
        for word_sentence, label_sentence in zip(word_sentence_list, label_sentence_list):
            entity_sentence = _get_entity_set(word_sentence, label_sentence)
            entity_sentence_list.append(entity_sentence)
            
        return entity_sentence_list
        
def _load_embedding(embedding_dir):
    token_file = os.path.join(embedding_dir, "token_list.npy")
    token_list = np.load(token_file)
    
    vector_file = os.path.join(embedding_dir, "vector_list.npy")
    vector_list = np.load(vector_file)
    
    token_to_vector = dict(zip(token_list, vector_list))
    d = vector_list.shape[1]
    return token_to_vector, d
    
def _read_pos_list(label_list_file):
    with open(label_list_file, "r") as f:
        line_list = f.read().splitlines()
        
    label_list = []
    for line in line_list:
        label, count = line.split(" ")
        label_list.append(label)
    label_to_index = {label: index for index, label in enumerate(label_list)}
    
    return label_list, label_to_index
    
def _read_entity_type_list(label_list_file):
    with open(label_list_file, "r") as f:
        line_list = f.read().splitlines()
        
    label_list = ["O"]
    for line in line_list:
        label, count = line.split(" ")
        label_list.append(label+":S")
        label_list.append(label+":B")
        label_list.append(label+":I")
        label_list.append(label+":E")
    label_to_index = {label: index for index, label in enumerate(label_list)}
    
    return label_list, label_to_index
    
def _normalize_sentence(raw_sentence):
    normal_string_list = []
    normal_to_raw_index = []
    
    for raw_index, raw_string in enumerate(raw_sentence):
        normal_string = unicodedata.normalize("NFKD", raw_string)
        normal_string_list.append(normal_string)
        normal_to_raw_index.append(raw_index)
        
        for _ in normal_string[1:]:
            normal_to_raw_index.append(-1)
            
    normal_sentence = "".join(normal_string_list)
    return normal_sentence, normal_to_raw_index
    
def _segment_sentence(sentence, delimiter_set):
    if not sentence:
        return [""]
        
    segment_list = []
    segment = ""
    for c in sentence:
        segment += c
        if c in delimiter_set:
            segment_list.append(segment)
            segment = ""
    if segment:
        segment_list.append(segment)
    return segment_list
    
def _segment_word_sentence(sentence, delimiter_set):
    if not sentence:
        return [[]]
        
    segment_list = []
    segment = []
    for word in sentence:
        segment.append(word)
        if word in delimiter_set:
            segment_list.append(segment)
            segment = []
    if segment:
        segment_list.append(segment)
    return segment_list
    
def _get_ws_batch_list(sentence_list, batch_sentences, batch_characters):
    index_sentence_list = sorted(
        enumerate(sentence_list),
        key = lambda index_sentence: len(index_sentence[1])
    )
    
    empty_sentence_batch = []
    for index, sentence in index_sentence_list:
        if len(sentence) > 0: break
        empty_sentence_batch.append((index, (sentence,[])))
    empty_sentences = len(empty_sentence_batch)
    if empty_sentence_batch:    
        batch_list = [empty_sentence_batch]
    else:
        batch_list = []
        
    batch = []
    for index, sentence in index_sentence_list[empty_sentences:]:
        batch.append((index, (sentence,["B" for c in sentence])))
        if len(batch)>=batch_sentences or len(batch)*len(batch[-1][1][0])>=batch_characters:
            batch_list.append(batch)
            batch = []
    if batch:
        batch_list.append(batch)
        
    return batch_list
    
def _get_pos_batch_list(sentence_list, batch_sentences, batch_characters):
    index_sentence_list = sorted(
        enumerate(sentence_list),
        key = lambda index_sentence: sum(len(word) for word in index_sentence[1])
    )
    
    empty_sentence_batch = []
    for index, sentence in index_sentence_list:
        if len(sentence) > 0: break
        empty_sentence_batch.append((index, (sentence,[])))
    empty_sentences = len(empty_sentence_batch)
    if empty_sentence_batch:    
        batch_list = [empty_sentence_batch]
    else:
        batch_list = []
        
    batch = []
    for index, sentence in index_sentence_list[empty_sentences:]:
        batch.append((index, (sentence,["D" for word in sentence])))
        if len(batch)>=batch_sentences or len(batch)*sum(len(word) for word in batch[-1][1][0])>=batch_characters:
            batch_list.append(batch)
            batch = []
    if batch:
        batch_list.append(batch)
        
    return batch_list
    
def _get_ner_batch_list(word_sentence_list, pos_sentence_list, batch_sentences, batch_characters):
    label_sentence_list = [
        [
            "O"
            for word in word_sentence
                for character in word
        ]
        for word_sentence in word_sentence_list
    ]
    
    index_sample_list = sorted(
        enumerate(zip(word_sentence_list, pos_sentence_list, label_sentence_list)),
        key = lambda i_wpl: sum(len(word) for word in i_wpl[1][0])
    )
    
    empty_sentence_batch = []
    for index, sample in index_sample_list:
        if len(sample[2]) > 0: break
        empty_sentence_batch.append((index, sample))
    empty_sentences = len(empty_sentence_batch)
    if empty_sentence_batch:
        batch_list = [empty_sentence_batch]
    else:
        batch_list = []
        
    batch = []
    for index, sample in index_sample_list[empty_sentences:]:
        batch.append((index, sample))
        if len(batch)>=batch_sentences or len(batch)*sum(len(word) for word in batch[-1][1][0])>=batch_characters:
            batch_list.append(batch)
            batch = []
    if batch:
        batch_list.append(batch)
        
    return batch_list
    
def _get_word_sentence_from_seq_sentence(sentence, seq_sentence):
    assert len(sentence) == len(seq_sentence)
    if not sentence:
        return []
        
    word_sentence = []
    word = sentence[0]
    for character, label in zip(sentence[1:], seq_sentence[1:]):
        if label == "B":
            word_sentence.append(word)
            word = ""
        word += character
    word_sentence.append(word)
    
    return word_sentence
    
def _get_forced_chunk_set(sentence, length_word_weight):
    
    chunk_to_weight = {}
    
    for i in range(len(sentence)):
        for length, word_to_weight in length_word_weight:
            word = sentence[i:i+length]
            if word in word_to_weight:
                chunk_to_weight[(i, i+length)] = word_to_weight[word]
                
    chunk_set = set()
    empty_sentence = [True] * len(sentence)
    
    for (l, r), w in sorted(chunk_to_weight.items(), key=lambda x: (x[1], x[0][1]-x[0][0]), reverse=True):
        empty = True
        for i in range(l, r):
            if not empty_sentence[i]:
                empty = False
                break
        if not empty: continue
        
        chunk_set.add((l,r))
        for i in range(l, r):
            empty_sentence[i] = False
            
    return chunk_set
    
def _soft_force_seq_sentence(forced_chunk_set, seq_sentence):
    for l, r in sorted(forced_chunk_set):
        if seq_sentence[l] != "B": continue
        if r<len(seq_sentence) and seq_sentence[r] != "B": continue
        for i in range(l+1,r):
            seq_sentence[i] = "I"
    return
    
def _hard_force_seq_sentence(forced_chunk_set, seq_sentence):
    for l, r in sorted(forced_chunk_set):
        seq_sentence[l] = "B"
        for i in range(l+1,r):
            seq_sentence[i] = "I"
        if r < len(seq_sentence):
            seq_sentence[r] = "B"
    return
    
def _run_word_segmentation_with_dictionary(word_sentence, recommend_dictionary, coerce_dictionary):
    sentence = "".join(word_sentence)
    seq_sentence = []
    for word in word_sentence:
        seq_sentence.append("B")
        for character in word[1:]:
            seq_sentence.append("I")
            
    recommend_chunk_set = _get_forced_chunk_set(sentence, recommend_dictionary)
    _soft_force_seq_sentence(recommend_chunk_set, seq_sentence)
    coerce_chunk_set = _get_forced_chunk_set(sentence, coerce_dictionary)
    _hard_force_seq_sentence(coerce_chunk_set, seq_sentence)
    
    word_sentence = _get_word_sentence_from_seq_sentence(sentence, seq_sentence)
    return word_sentence
    
def _force_whitespace_tagging(word_sentence, pos_sentence):
    for i, word in enumerate(word_sentence):
        if _whitespace_pattern.fullmatch(word) is not None:
            pos_sentence[i] = "WHITESPACE"
    return
    
def _get_entity_set(word_sentence, label_sentence):
    sentence = "".join(word_sentence)
    assert len(sentence) == len(label_sentence)
    
    entity_set = set()
    l = None
    for i, label in enumerate(label_sentence):
        if label == "O":
            l = None
        else:
            entity_type, seq_type = label.split(":")
            if seq_type == "S":
                entity_set.add(
                    (i, i+1, entity_type, sentence[i:i+1])
                )
                l = None
            elif seq_type == "B":
                l = i
            elif seq_type == "E":
                if l is not None:
                    entity_set.add(
                        (l, i+1, entity_type, sentence[l:i+1])
                    )
                    l = None
    return entity_set
    
def main():
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    
