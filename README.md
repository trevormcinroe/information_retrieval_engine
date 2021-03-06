```
          _____                    _____                    _____ 
         /\    \                  /\    \                  /\    \         
        /::\    \                /::\    \                /::\    \        
        \:::\    \              /::::\    \              /::::\    \       
         \:::\    \            /::::::\    \            /::::::\    \      
          \:::\    \          /:::/\:::\    \          /:::/\:::\    \     
           \:::\    \        /:::/__\:::\    \        /:::/__\:::\    \    
           /::::\    \      /::::\   \:::\    \      /::::\   \:::\    \   
  ____    /::::::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \  
 /\   \  /:::/\:::\    \  /:::/\:::\   \:::\____\  /:::/\:::\   \:::\    \ 
/::\   \/:::/  \:::\____\/:::/  \:::\   \:::|    |/:::/__\:::\   \:::\____\
\:::\  /:::/    \::/    /\::/   |::::\  /:::|____|\:::\   \:::\   \::/    /
 \:::\/:::/    / \/____/  \/____|:::::\/:::/    /  \:::\   \:::\   \/____/ 
  \::::::/    /                 |:::::::::/    /    \:::\   \:::\    \     
   \::::/____/                  |::|\::::/    /      \:::\   \:::\____\    
    \:::\    \                  |::| \::/____/        \:::\   \::/    /    
     \:::\    \                 |::|  ~|               \:::\   \/____/     
      \:::\    \                |::|   |                \:::\    \         
       \:::\____\               \::|   |                 \:::\____\        
        \::/    /                \:|   |                  \::/    /        
         \/____/                  \|___|                   \/____/         
```

![alt-text](./src/assets/screencast.gif)

### Introduction         
The Information Retrieval Engine (IRE) is a CLI tool for finding relevant data that is powered by two deep learning models, a keyword extraction algorithm, and MongoDB. It was created as a two-part final project for Northwestern's MSDS-453 Natural Language Processing course. 

### How does it work?
As a preprocessing step, 10,000 Wikipedia pages were summarized with (1) and had their keywords extracted with (3).
This data was then stored in a local instance of MongoDB. During runtime, a user provides a keyword-based query. These keywords
are used to filter the database to a set of summary candidates. The keywords and candidate summaries are then tokenized and embedded with (4) and fed through 
model (2) to determine their semantic similarity. Finally, the summary with the highest predicted similarity is returned to the user.

### Models/Algorithms
1. BART pre-trained on the CNN/Daily Mail dataset for article summarization
    * [BART: Denoising Sequence-to-Dequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)
2. GRU Siamese network trained on the SemEval-2014 dataset for semantic similarity
    * [Learning a Similarity Metric Discriminatively, with Application to Face Verification](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)
3. TextRank algorithm
    * [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
4. Word2Vec embeddings pre-trained on the Google News corpus
    * [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

