from flask import Flask, render_template, request
import tensorflow as tf
import pickle as pkl
from pyvi import ViTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
with open('tokenizer_data.pkl', 'rb') as file:
    my_tokenizer = pkl.load(file)
my_model = load_model('CNN-BILSTM_model.h5', compile=False)
maxlen_vector = 600

def get_vectorize(input, tokenizer):
    input_text_pre = list(tf.keras.preprocessing.text.text_to_word_sequence(input))
    input_text_pre = ' '.join(input_text_pre)
    input_text_pre_accent = ViTokenizer.tokenize(input_text_pre)
    tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre_accent])
    vec_data = pad_sequences(tokenized_data_text, padding='post', maxlen=600)
    return vec_data

def get_confidence(feature, model):
    label_dict = {'negative': 0, 'positive': 1}
    label = list(label_dict.keys())
    output = model(feature).numpy()[0]
    result = output.argmax()
    conf = round(float(output.max()), 4) * 100
    return label[int(result)], conf

def get_prediction(input, tokenizer, model):
    input_model = get_vectorize(input, tokenizer)
    result, conf = get_confidence(input_model, model)
    return result, conf

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    confidence = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input:
            sentiment_result, confidence = get_prediction(user_input, my_tokenizer, my_model)
    return render_template('home.html', sentiment_result=sentiment_result, confidence=(confidence))

if __name__ == '__main__':
    app.run(debug=True)