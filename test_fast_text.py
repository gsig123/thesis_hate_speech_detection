from pyfasttext import FastText

model_path = "./models/fast_text/OffensEval_EN.bin"
model = FastText(model_path)
print(model['@USER'])
print(len(model['@USER']))
