# ckipneutools

This open-source library implements:
* Neural CKIP-style Chinese word segmentation
* Neural CKIP-style Chinese part-of-speech tagging
* Neural CKIP-style Chinese named entity recognition

Neu tools demo: [neu](http://ckip.iis.sinica.edu.tw/service/ckipneutools)<br />
Old tools demo: [old](http://ckip.iis.sinica.edu.tw/service/corenlp)

Performance Neu vs. Old
- WS: +1.4% absolute F1 on ASBC 4.0 test split
- POS: +4.0% absolute accuracy on ASBC 4.0 test split
- NER: +2.2% absolute F1 on OntoNotes 5.0 Chinese test split

Ease-of-use Neu vs. Old
- Do not auto delete/change/add characters
- Keep spaces as they are
- Keep full/half-width characters as they are
- Do not auto insert newlines
- Support indefinitely long sentences

Features Neu
- Do not rely on word list, word frequency statisics, PoS frequency statisics
- Support user-defined recommended-word list
- Support user-defined must-word list

## Installation

tl;dr.
```
pip install ckipneutools[tf,gdown]
```

ckipneutools is a Python library hosted on PyPI. Requirements:
- python>=3.6
- tensorflow / tensorflow-gpu (one-of-them)
- gdown (optional, for downloading model files from google drive)

(Minimum installation) If you have set up tensorflow, and would like to download model files by your self.
```
pip install ckipneutools
```

(Complete installation) If you have just set up a clean virtual environment, and want everything, including GPU support.
```
pip install ckipneutools[tfgpu,gdown]
```

## Usage

See the complete demo script: demo.py<br />
Or the [web demo](http://ckip.iis.sinica.edu.tw/service/ckipneutools)

### 1. Download model files

The model files are available on [google drive](https://drive.google.com/drive/folders/15BDjL2IaX3eYdFVzT422VwCb743Hrbi3). If you have gdown installed, you can download and extract to desired path by the included API.
```python
ckipneutools.data_utils.download_data("./")
```

### 2. Load model
```python
ws = ckipneutools.WS("./data")
pos = ckipneutools.POS("./data")
ner = ckipneutools.NER("./data")
```

### 3. (Optional) Create dictionary

You can supply words for WS speicial consideration, including their relative weights.
```python
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
```
```
[(2, {'公有': 2.0}), (3, {'土地公': 1.0, '土地婆': 1.0}), (5, {'緯來體育台': 1.0})]
```

### 4. Run the WS-POS-NER pipeline
```python
sentence_list = [
    "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
    "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
    "",
    "土地公有政策?？還是土地婆有政策。.",
    "… 你確定嗎… 不要再騙了……",
    "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
    "科長說:1,坪數對人數為1:3。2,可以再增加。",
]

word_sentence_list = ws(
    sentence_list,
    # sentence_segmentation=True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced
)

pos_sentence_list = pos(word_sentence_list)

entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
```

### 5. (Optional) Release memory
```python
del ws
del pos
del ner
```

### 6. Show Results
```python
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
```
```

'傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'
傅達仁(Nb)　今(Nd)　將(D)　執行(VC)　安樂死(Na)　，(COMMACATEGORY)　卻(D)　突然(D)　爆出(VJ)　自己(Nh)　20(Neu)　年(Nf)　前(Ng)　遭(P)　緯來(Nb)　體育台(Na)　封殺(VC)　，(COMMACATEGORY)　他(Nh)　不(D)　懂(VK)　自己(Nh)　哪裡(Ncd)　得罪到(VJ)　電視台(Nc)　。(PERIODCATEGORY)　
(0, 3, 'PERSON', '傅達仁')
(18, 22, 'DATE', '20年前')
(23, 28, 'ORG', '緯來體育台')

'美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。'
美國(Nc)　參議院(Nc)　針對(P)　今天(Nd)　總統(Na)　布什(Nb)　所(D)　提名(VC)　的(DE)　勞工部長(Na)　趙小蘭(Nb)　展開(VC)　認可(VC)　聽證會(Na)　，(COMMACATEGORY)　預料(VE)　她(Nh)　將(D)　會(D)　很(Dfa)　順利(VH)　通過(VC)　參議院(Nc)　支持(VC)　，(COMMACATEGORY)　成為(VG)　該(Nes)　國(Nc)　有史以來(D)　第一(Neu)　位(Nf)　的(DE)　華裔(Na)　女性(Na)　內閣(Na)　成員(Na)　。(PERIODCATEGORY)　
(0, 2, 'GPE', '美國')
(2, 5, 'ORG', '參議院')
(7, 9, 'DATE', '今天')
(11, 13, 'PERSON', '布什')
(17, 21, 'ORG', '勞工部長')
(21, 24, 'PERSON', '趙小蘭')
(42, 45, 'ORG', '參議院')
(56, 58, 'ORDINAL', '第一')
(60, 62, 'NORP', '華裔')

''


'土地公有政策?？還是土地婆有政策。.'
土地公(Nb)　有(V_2)　政策(Na)　?(QUESTIONCATEGORY)　？(QUESTIONCATEGORY)　還是(Caa)　土地(Na)　婆(Na)　有(V_2)　政策(Na)　。(PERIODCATEGORY)　.(PERIODCATEGORY)　
(0, 3, 'PERSON', '土地公')

'… 你確定嗎… 不要再騙了……'
…(ETCCATEGORY)　 (WHITESPACE)　你(Nh)　確定(VK)　嗎(T)　…(ETCCATEGORY)　 (WHITESPACE)　不要(D)　再(D)　騙(VC)　了(Di)　…(ETCCATEGORY)　…(ETCCATEGORY)　

'最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.'
最多(VH)　容納(VJ)　59,000(Neu)　個(Nf)　人(Na)　,(COMMACATEGORY)　或(Caa)　5.9萬(Neu)　人(Na)　,(COMMACATEGORY)　再(D)　多(D)　就(D)　不行(VH)　了(T)　.(PERIODCATEGORY)　這(Nep)　是(SHI)　環評(Na)　的(DE)　結論(Na)　.(PERIODCATEGORY)　
(4, 10, 'CARDINAL', '59,000')
(14, 18, 'CARDINAL', '5.9萬')

'科長說:1,坪數對人數為1:3。2,可以再增加。'
科長(Na)　說(VE)　:1,(Neu)　坪數(Na)　對(P)　人數(Na)　為(VG)　1:3(Neu)　。(PERIODCATEGORY)　2(Neu)　,(COMMACATEGORY)　可以(D)　再(D)　增加(VHC)　。(PERIODCATEGORY)　
(4, 6, 'CARDINAL', '1,')
(12, 13, 'CARDINAL', '1')
(14, 15, 'CARDINAL', '3')
(16, 17, 'CARDINAL', '2')

```

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
Copyright 2019 CKIP
