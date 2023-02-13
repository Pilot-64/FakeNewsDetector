from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model('fakenewsmodel.h5')

# model evaluation
input = "Kerry to go to Paris in gesture of sympathy"
  
tokenizer1 = Tokenizer()

# detection
sequences = tokenizer1.texts_to_sequences([input])[0]
sequences = pad_sequences([sequences], maxlen=54,
                          padding='post', 
                          truncating='post')
if(model.predict(sequences, verbose=0)[0][0] >= 0.5):
    print("This news is True")
else:
    print("This news is false")