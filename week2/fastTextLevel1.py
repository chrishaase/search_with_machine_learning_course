import fasttext

# Train model standard
# model = fasttext.train_supervised(input="../../datasets/fasttext/products.train")
# model.save_model("product_model_standard.bin")

# train with 25 epochs, bigrams, and learning rate of 1.0
# model = fasttext.train_supervised(input="../../datasets/fasttext/products.train", lr=1.0, epoch=25, wordNgrams=2)
# model.save_model("product_model_e25_lr1_wng2.bin")

# train with 25 epochs, bigrams, and learning rate of 1.0
# model = fasttext.train_supervised(input="../../datasets/fasttext/products_normalized.train", lr=1.0, epoch=25, wordNgrams=2)
# model.save_model("product_normalized_model_e25_lr1_wng2.bin")

# train with 25 epochs, bigrams, and learning rate of 1.0
model = fasttext.train_supervised(input="../../datasets/fasttext/products_normalized_maxcat.train", lr=1.0, epoch=25, wordNgrams=2)
model.save_model("product_normalized_maxcat_model_e25_lr1_wng2.bin")

# Load model
# model = fasttext.load_model("product_model_standard.bin")
# model = fasttext.load_model("product_model_e25_lr1_wng2.bin")
# model = fasttext.load_model("product_normalized_model_e25_lr1_wng2.bin")
# model = fasttext.load_model("product_normalized_maxcat_model_e25_lr1_wng2.bin")

# Test single prediction
# result = model.predict("Alvarez - Regent 6-String Full-Size Cutaway Dreadnought Acoustic/Electric Guitar - Natural")
# print(result)

# Evaluate on test data
# result = model.test("../../datasets/fasttext/products.test")
# print(result)

# Test single prediction normalized
result = model.predict("alvarez regent 6 string full size cutaway dreadnought acoustic electric guitar natural")
print(result)

# Evaluate on normalized test data
result = model.test("../../datasets/fasttext/products_normalized_maxcat.test")
print(result)


