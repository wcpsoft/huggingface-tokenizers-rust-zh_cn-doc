# 快速入门
让我们快速了解一下 🤗 Tokenizers库的功能。该库提供了当今最常用的分词器的实现，不仅易于使用，而且速度极快。
## 从零开始构建一个属于自己分词器
为了展示 🤗 Tokenizers 库的速度有多快，我们将在几秒钟内对 wikitext-103（516M 的文本）训练一个新的分词器。首先，你需要下载这个数据集并解压它：
```
wget https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/wikitext/wikitext-103-raw-v1.zip

unzip wikitext-103-raw-v1.zip
```
## 构建分词器
在这个教程中，我们将构建并训练一个字节对编码（BPE）分词器。
在这个分词器中你将学习以下知识：
- 如何通过训练语料库生成令牌
- 如何识别最常见的令牌对并根据规则合并令牌，并且重复此合并过程，直到完成我们想要词汇表（例如满足某个数量级的令牌）
### 创建一个RUST项目

### 安装库文件
可通过Crates.io库安装Tokenizers库，需配置项目中的Cargo.toml文件
```
tokenizers = "0.10"
```

### 第一步 
在当前分词器实现过程中，主要会用到Tokenizer以下API，我们将使用BPE模型算法进行举例说明：
```rust
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TokenizerImpl};

let mut tokenizer: TokenizerImpl<
    BPE,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
> = TokenizerImpl::new(
    BPE::builder()
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap(),
);
```
为了训练wikitext-103数据集，首先需要构建一个BpeTrainer。
接下我们需要设置一些训练参数，例如vocab_size（默认值：30,000）和min_frequency（默认值：0）等，但最重要的是指定我们特殊token列表规范，使用的special_tokens将特殊token列表（例如：[UNK]）按照顺序添加到词汇表中。
特殊token列表会根据special_tokens中数组的顺序进行插入，所以这边插入顺序很重要。
例如：
- `[UNK]` ID 将在词汇表中设置为 0
- `[CLS]` ID 将在词汇表中设置为 1
- 以此类推。
```rust
use tokenizers::models::bpe::BpeTrainer;
let mut trainer = BpeTrainer::builder()
    .special_tokens(vec![
        AddedToken::from("[UNK]", true),
        AddedToken::from("[CLS]", true),
        AddedToken::from("[SEP]", true),
        AddedToken::from("[PAD]", true),
    ])
    .build();
```
构建好训练器后，我们可以使用它来训练我们的分词器，但这样并不是一个好的训练规则，所以我们将设置一些训练语料的预处理规则，当前示例中我们将做个最简单的规则，即按空格分割分割语料。
如不设定预处理步骤，将可能会得到“it is”这样的令牌，因为他们是高频相邻出现的词。
```rust
use tokenizers::pre_tokenizers::whitespace::Whitespace;
tokenizer.with_pre_tokenizer(Whitespace {});
```
现在，我们可以使用wikitext-103数据集中的任何文件来调用 `Tokenizer.train` 方法进行训练。只需要几秒钟就能在完整的wikitext数据集上训练我们的分词器！
```rust
let files = vec![
    "data/wikitext-103-raw/wiki.train.raw".into(),
    "data/wikitext-103-raw/wiki.test.raw".into(),
    "data/wikitext-103-raw/wiki.valid.raw".into(),
];
tokenizer.train_from_files(&mut trainer, files)?;
```
接下来要将分词器保存到一个包含其所有配置和词汇表的文件中，只需使用 `Tokenizer.save` 方法,这样就可以将训练好的分词器保存到tokenizer.json 文件中,然后，你可以使用 Tokenizer.from_file类方法从该文件重新加载你的分词器,并使用tokenizer进行测试。
现在我们可以使用Tokenizer.encode方法对任何文本进行分词进行测试工作。
```rust
tokenizer.save("data/tokenizer-wiki.json", false)?;
let mut tokenizer = Tokenizer::from_file("data/tokenizer-wiki.json")?;
let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
```
目前，我们已经完成训练一个简单分词器的完整流程及工作，它可以将任何文本进行分词并返回一个Encoding对象，通过Encoding的向量组我们可做相似度分析以及transformer模型的输入等。当然这个分词器离正在好用还有很多优化点，因为它过于简单向量词表不够紧凑等问题可能造成算法应用的效果不理想。

在Encoding对象中我们可找到关于深度学习（或其他应用）所需的所有属性方法。
获取Encoding对象属性方法如下：
```rust
println!("{:?}", output.get_tokens());
```
具体对象结构如下：
- get_tokens 获取分词结果
- get_ids 获取token结果，即每个分词结果在分词器词汇表中的索引向量表示，如[220,22,12]。
- get_offsets 获取索引获取分词结果在原文中的位置，如[(0, 5), (6, 8), (9, 12)]。
- get_type_ids 获取每个分词结果在原文中的类型，如[0,0,1]。
- get_attention_mask 获取掩码位置标识，这在检查注意力学习掩码处理时很有用。

接下来我们可以添加后置处理步骤，我们可能希望分词器自动添加特殊令牌，如 [CLS] 或 [SEP]，以便于我们完成后续的任务。TemplateProcessing是最常用的后处理器，你只需指定单个句子和句子对的处理模板，以及特殊令牌及其ID。

在构建我们的分词器时，我们已经在特殊令牌列表的第 1 和第 2 位置设置了 [CLS] 和 [SEP]，我们可通过Tokenizer.token_to_id来检测生成词表位置ID是否是我们期望的，可使用以下方法：
```rust
println!("{}", tokenizer.token_to_id("[SEP]").unwrap());
```
