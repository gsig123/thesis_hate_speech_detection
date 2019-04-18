./fastText/fasttext print-word-vectors fast_text_pre_trained/cc.en.300.bin < ./EN_tokens_OOV.txt > ./EN_tokens_OOV_embeddings.txt &&
./fastText/fasttext print-word-vectors fast_text_pre_trained/cc.da.300.bin < ./DA_1600_tokens_OOV.txt > ./DA_1600_OOV_embeddings.txt
