# 用CRF完成NER和POS Tagging

该directory下的代码基于Sameer Singh的Statistical Natural Language Processing HW3


在运行代码之前，你需要先安装[allennlp](https://github.com/allenai/allennlp)库

```
pip install allennlp
```

然后你可以运行以下代码
```
allennlp train [JSON file path] -s [save point file path] --include-package neural_crf
```


E.g., `simple_tagger` model for POS
```
allennlp train ./config/simple_tagger_pos.json -s ./model/simple_tagger.pt
```
E.g., `neural_crf` model for POS
```
allennlp train ./config/neural_crf.json -s ./model/neural_crf_pos.pt --include-package neural_crf
```

Sameer Singh的原始课程作业描述[在此](https://canvas.eee.uci.edu/courses/14385/assignments/270636).

## 文件

该文件夹下有以下代码:


* `viterbi.py`: 实现了基于CRF模型的viterbi decoder。运行`python viterbi_test.py`应该可以通过所有测试。 

### 你可以尝试修改以下代码
* `config/simple_tagger_{pos,ner}.json`: `simple_tagger` 模型的配置代码，可以用来训练allennlp实现的简单POS和NER模型。

* `config/neural_crf_{pos,ner}.json`: `neural_crf`模型的配置代码，可以用来训练基于CRF的POS和NER模型。

### 以下代码建议不要修改

* `neural_crf.py`: Neural CRF 模型，调用了`viterbi.py`。

* `viterbi_test.py`: 测试Viterbi算法的代码。
