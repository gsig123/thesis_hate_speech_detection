# Creating Tokens + Embeddings

## Step 1
- Run `train_file_to_token_file.py` for the train files you want to create tokens for.
- This creates tokens in the same way as our `pre_trained_embedding_pipeline` and saves them to a file.
- Example:
```sh
python train_file_to_token_file.py ./data/raw/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv EN_tokens.txt

python train_file_to_token_file.py ./data/raw/OffensEval2019_Danish/danish_1600.tsv  DA_1600_tokens.txt
```

## Step 2
- Run `create_embedding_files.py`.
- Make sure the constants in that file are correct.
- This goes through the token files, tries to find the embeddings from the pre-trained FastText embeddings for each language (top 100k most frequent) and saves these to a file.
- The tokens that don't have embeddings int he pre-trained embeddings are saved to file.
- Example:
```sh
python create_embedding_files.py
```

## Step 3
- Run `tokens_oov_emb.sh`.
- This generates embeddings for the OOV tokens and saves to file. 
- Example:
```sh
sh ./tokens_oov_emb.sh
```

## Step 4
- Run `combine_embedding_files.sh`
- This combines OOV and found embeddings into files
- Example:
```sh
sh ./combine_embedding_files.sh
```