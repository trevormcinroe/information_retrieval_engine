HEADER = r"""          _____                    _____                    _____          
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
Information Retrieval Engine
author: trevor mcinroe
"""

about_msg = """\nThe Information Retrieval Engine was created by Trevor McInroe as a two-part project for Northwestern University's MSDS-453 Natural Language Processing class. It has five main components: a keyword extraction algorithm (TextRank), a text summarization model (BART encoder-decoder with Attention & Beam search), a semantic similarity model (GRU Siamese network), word2vec embeddings, and MongoDB. 10,000 Wikipedia articles were summarized and had their keywords extracted. This data is stored on a locally-running MongoDB instance. When a user makes a query, the DB is filtered to documents with similar keywords. The summaries of this filtered list are fed, along with the query, into the Siamese network and the summary with the highest predicted semantic similarity is returned."""