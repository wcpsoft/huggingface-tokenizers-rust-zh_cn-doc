# 分词器执行流程介绍
当调用 Tokenizer.encode 或 Tokenizer.encode_batch 时，输入文本会经过以下流程，将通过quickstart[./quickstart.md]的示例来说明各流程。
首先我们将quickstart的分词器加载运行起来
```rust
use tokenizers::Tokenizer;
let mut tokenizer = Tokenizer::from_file("data/tokenizer-wiki.json")?;
```
## 规范化（Normalization）
在规范化阶段就是将需要学习的语料原始文本进行统一化的处理，使其更加干净一致，让学习后的结果更符合算法预期。常见的操作包括去除空格、特殊符号、全部文本转为小写、或者用Unicode或UTF-8编码集来标识文本内容，这些都是大多数分词器的规范化操作，可根据自己算进行调整。
在🤗 Tokenizers 库中，每个规范化操作都由一个 Normalizer组件负责，你可以使用 normalizers.Sequence 来组合多个规范化操作。在Tokenizers的处理过程中，在规范化过程中会保留对齐信息，为了能够在生成的令牌与原始输入文本之间进行映射。
当然你是否使用规范化阶段是可选的根据自己算法模式需要。
| 名称           | 描述                                                                                                               | 示例                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| NFD            | NFD Unicode 规范化                                                                                                 |                                                                       |
| NFKD           | NFKD Unicode 规范化                                                                                                |                                                                       |
| NFC            | NFC Unicode 规范化                                                                                                 |                                                                       |
| NFKC           | NFKC Unicode 规范化                                                                                                |                                                                       |
| Lowercase      | 将所有大写字母转换为小写字母                                                                                       | 输入: HELLO ὈΔΥΣΣΕΎΣ<br>输出: hello ὀδυσσεύς                          |
| Strip          | 移除输入字符串指定方向（左、右或两侧）的所有空白字符                                                               | 输入: " hi "<br>输出: "hi"                                            |
| StripAccents   | 移除 Unicode 中的所有重音符号（通常与 NFD 结合使用以确保一致性）                                                   | 输入: é<br>输出: e                                                    |
| Replace        | 替换自定义字符串或正则表达式的匹配内容为给定的文本                                                                 | Replace("a", "e") 将执行如下操作:<br>输入: "banana"<br>输出: "benene" |
| BertNormalizer | 提供原始 BERT 使用的 Normalizer 的实现。可设置的选项有: clean_text, handle_chinese_chars, strip_accents, lowercase |                                                                       |
| Sequence       | 组合多个规范化器，以指定的顺序依次运行                                                                             | Sequence::new(vec![NFKC, Lowercase])                                  |


下面我们以西欧语系的规范化操作为例说明，主要包含两个步骤：NFD Unicode规范化和去除重音
```rust
use tokenizers::normalizers::{
    strip::StripAccents, unicode::NFD, utils::Sequence as NormalizerSequence,
};
let normalizer = NormalizerSequence::new(vec![NFD.into(), StripAccents.into()]);
let mut normalized = NormalizedString::from("Héllò hôw are ü?");
normalizer.normalize(&mut normalized)?;
println!("{}", normalized.get());
```
## 预分词（Pre-tokenization）
在预分词阶段，将文本按照一定的规则将预料料切分为多个单词或句子，让其符合模型输入要求。最简单预分词方式就是按照标点符号进行切分，例如中文的标点符号，英文的标点符号等，当然你可以使用Sequence来实现多个预分词方法的组合。
| 名称               | 描述                                                                                                                                                                                                                                                                                                                                                                       | 示例                                                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| ByteLevel          | 基于字节进行分割，同时将所有字节映射为可见字符集。此技术由 OpenAI 在 GPT-2 中引入，具有以下特点：<br>由于它基于字节，因此仅需 256 个字符作为初始字母表（字节的取值范围为 256），而不是 130,000+ 个 Unicode 字符。<br>因此，使用此方法时完全不需要未知 Token，因为我们可以用 256 个 Token 表示任何内容（太棒了🎉🎉）。<br>对于非 ASCII 字符，它会变得完全不可读，但仍然有效！ | 输入: "Hello my friend, how are you?"<br>输出: "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġare", "Ġyou", "?"                                 |
| Whitespace         | 在单词边界处分割（使用正则表达式 \w+                                                                                                                                                                                                                                                                                                                                       | [^\w\s]+）                                                                                                                                 | 输入: "Hello there!"<br>输出: "Hello", "there", "!" |
| WhitespaceSplit    | 在任何空白字符处进行分割                                                                                                                                                                                                                                                                                                                                                   | 输入: "Hello there!"<br>输出: "Hello", "there!"                                                                                            |
| Punctuation        | 将所有标点符号隔离开来                                                                                                                                                                                                                                                                                                                                                     | 输入: "Hello?"<br>输出: "Hello", "?"                                                                                                       |
| Metaspace          | 在空白字符处分割，并将其替换为特殊字符“▁” (U+2581)                                                                                                                                                                                                                                                                                                                         | 输入: "Hello there"<br>输出: "Hello", "▁there"                                                                                             |
| CharDelimiterSplit | 在指定字符处分割。以字符“x”为例：                                                                                                                                                                                                                                                                                                                                          | 输入: "Helloxthere"<br>输出: "Hello", "there"                                                                                              |
| Digits             | 将数字与其他字符分开                                                                                                                                                                                                                                                                                                                                                       | 输入: "Hello123there"<br>输出: "Hello", "123", "there"                                                                                     |
| Split              | 多功能的预分词器，可以根据提供的模式和行为进行分割。模式可以是自定义字符串或正则表达式，行为可以是以下之一：<br>Removed<br>Isolated<br>MergedWithPrevious<br>MergedWithNext<br>Contiguous<br>invert 是一个布尔标志。                                                                                                                                                       | 示例，模式 = ","，行为 = "isolated"，invert = False:<br>输入: "Hello, how are you?"<br>输出: "Hello,", " ", "how", " ", "are", " ", "you?" |
| Sequence           | 允许你组合多个 PreTokenizer 按指定顺序运行                                                                                                                                                                                                                                                                                                                                 | Sequence::new(vec![Punctuation, WhitespaceSplit])                                                                                          |

通过预分词器 pre_tokenizers.Whitespace 来实现：
```rust
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
let pre_tokenizer = Whitespace {};
let mut pre_tokenized = PreTokenizedString::from("Hello! How are you? I'm fine, thank you.");
pre_tokenizer.pre_tokenize(&mut pre_tokenized)?;
println!(
    "{:?}",
    pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte)
);
// 结果大致如下：
// [("Hello", (0, 5), None), ("!", (5, 6), None), ("How", (7, 10), None),
//  ("are", (11, 14), None), ("you", (15, 18), None), ("?", (18, 19), None),
//  ("I", (20, 21), None), ("\'", (21, 22), None), ("m", (22, 23), None),
//  ("fine", (24, 28), None), (",", (28, 29), None), ("thank", (30, 35), None),
//  ("you", (36, 39), None), (".", (39, 40), None)]
```
## 模型（Model）
在文本数据经过规范化和预分词两个阶段处理后，🤗 Tokenizers 库会使用这些预处理后的数据进行分词模型的训练。训练过程中，不同的算法模型会根据预定的规则将文本切分为 Token（通常是子词或单词片段），并将这些 Token 映射到对应的词汇表中。训练完成后，相关的分词模型会被保存。
Tokenizers 库支持以下模型算法：
| 名称      | 描述                                                                                                                                                                                                                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| WordLevel | 这是“经典”的分词算法。它将单词直接映射到 ID，没有任何复杂的操作。其优点是使用简单且易于理解，但为了覆盖面好，需要非常大的词汇表。使用此模型时需要配合 PreTokenizer。此模型本身不会做出选择，而是简单地将输入的 Token 映射到 ID。                                                                             |
| BPE       | 这是最流行的子词分词算法之一。字节对编码（Byte-Pair-Encoding）通过从字符开始，逐步合并最常见的字符对，从而创建新的 Token。然后，它通过迭代的方式从语料库中最常见的字符对构建新的 Token。BPE 能够通过使用多个子词 Token 构建它从未见过的单词，因此需要较小的词汇表，并且减少了出现“unk” (未知 Token) 的概率。 |
| WordPiece | 这是一种与 BPE 相似的子词分词算法，主要由 Google 在 BERT 等模型中使用。它使用贪心算法，尽量先构建较长的单词，当整个单词不在词汇表中时，将其拆分为多个 Token。这与 BPE 从字符开始，构建更大 Token 的方式不同。它使用著名的 ## 前缀来标识属于单词的一部分的 Token（即不是单词的开始）。                        |
| Unigram   | Unigram 也是一种子词分词算法，它通过试图识别最佳的子词 Token 集合，最大化给定句子的概率。这与 BPE 不同，Unigram 并非通过一套规则顺序应用来决定，而是计算多种分词方式并选择最可能的一个。                                                                                                                 |

## 后处理（Post-processing）
在整个管道处理后，我们有时希望在将一个分词后的字符串输入到模型之前，插入一些特殊的 Token，例如“[CLS] My horse is amazing [SEP]”。PostProcessor 就是执行此操作的组件。
`tokenizers::processors::template::TemplateProcessing`: 这是一个后处理器，它允许你定义一个模板，用于处理输入的文本。这个模板可以包括特殊令牌，如 `[CLS]` 和 `[SEP]`，它们会被替换为对应的 ID。
```rust
use tokenizers::processors::template::TemplateProcessing;
tokenizer.with_post_processor(
    TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .unwrap()
        .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
        .unwrap()
        .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
        .build()
        .unwrap(),
);
```

## 解码器（Decoder）
在模型输出的 ID 序列中，我们通常需要将其解码为原始文本。例如，对于一个文本序列，我们可以将其解码为原始文本，以便在模型输出后，我们可以将其用作输入。
| 名称      | 描述                                                                                                                                                                        |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ByteLevel | 恢复 ByteLevel PreTokenizer 的操作。该 PreTokenizer 以字节级别进行编码，使用一组可见的 Unicode 字符表示每个字节，因此我们需要一个解码器来恢复这一过程并重新获得可读的内容。 |
| Metaspace | 恢复 Metaspace PreTokenizer 的操作。该 PreTokenizer 使用特殊标识符 ▁ 来标识空白字符，因此此解码器有助于解码这些字符。                                                       |
| WordPiece | 恢复 WordPiece 模型的操作。该模型使用特殊标识符 ## 来标识继续的子词，因此此解码器有助于解码这些子词。                                                                       |
```rust
let decoded = tokenizer.decode(output.get_ids(), true)?;
println!("{}", decoded);
````
