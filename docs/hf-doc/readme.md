<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    <a href="https://pepy.tech/project/tokenizers">
        <img src="https://pepy.tech/badge/tokenizers/week" />
    </a>
</p>

tokenizers是开源的分词库，提供了当今最常用的分词器的实现。

## 主要功能:

 - 使用当今最常用的分词器训练新的词汇表并进行分词。
 - 由于采用了Rust实现，以极快的速度完成包括训练和分词等功能，支持在设备上使用CPU进行处理，1GB文本所需时间不到20秒。
 - 易于使用，同时非常多功能。
 - 专为研究和生产设计。
 - 可以正则化过程中会跟踪对齐信息。始终可以获取对应于给定令牌的原始句子部分。
 - 可以完成所有预处理工作：截断、填充、添加模型所需的特殊令牌等。

## 性能
性能可能会根据硬件有所不同，但在g6 AWS实例上运行[~/bindings/python/benches/test_tiktoken.py](bindings/python/benches/test_tiktoken.py)应该会得到以下结果：
![image](https://github.com/user-attachments/assets/2b913d4b-e488-4cbc-b542-f90a6c40643d)


## 支持的语言

我们为以下语言提供了支持（还有更多语言即将支持）：
  - [Rust](https://github.com/huggingface/tokenizers/tree/main/tokenizers)（原生实现）
  - [Python](https://github.com/huggingface/tokenizers/tree/main/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/main/bindings/node)
  - [Ruby](https://github.com/ankane/tokenizers-ruby)（由@ankane贡献，外部仓库）
 
## 使用Python的快速入门：

选择Byte-Pair Encoding、WordPiece或Unigram任一一种模型完成分词器实例化：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
```

你可以自定义预分词的方式（例如，拆分成单词）：

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

然后在一组文件上训练你的分词器只需两行代码：

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
```

一旦你的分词器训练完成，只需一行代码即可编码任何文本：
```python
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```

详情查看[documentation](https://huggingface.co/docs/tokenizers/index)或 [quicktour](https://huggingface.co/docs/tokenizers/quicktour)进行更多学习