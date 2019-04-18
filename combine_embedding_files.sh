cat ./EN_tokens_embeddings.txt ./EN_tokens_OOV_embeddings.txt > ./EN_tokens_embeddings_combined.txt &&
cat ./DA_1600_tokens_embeddings.txt ./DA_1600_OOV_embeddings.txt > ./DA_tokens_embeddings_combined.txt &&
cat ./EN_tokens_embeddings_combined.txt ./DA_tokens_embeddings_combined.txt > ./EN_DA_tokens_embeddings_combined.txt