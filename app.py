import re, pickle
import pandas as pd
import sqlite3 as sql
import numpy as np
from flask import Flask, jsonify
from flask import request
from flask import Response
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



app =  Flask (__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = {
    'swagger': '2.0',
    'info' : {
        'title' :  'API Documentation for Data Processing and Predicting Sentiment',
        'description' : 'Dokumentasi API untuk processing dan modelling data teks untuk membersihkan kata-kata Hate Speech dan Abusive',
        'version' : '1.0.0'
        
    }

}
swagger_config = {
    'headers' : [],
    'specs' : [
        {
            'endpoint' : 'docs',
            'route' : '/docs.json'
        }
    ],
    'static_url_path' :'/flasgger_statis',
    'swagger_ui' :True,
    'specs_route' : '/docs/' 

}
swagger = Swagger(app, config= swagger_config, template=swagger_template)

connection_model = sql.connect("database.db", check_same_thread=False)
connection_data = sql.connect("database_sensoring.db", check_same_thread=False)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ',lower=True)
sentiment = ['negative', 'neutral', 'positive']

#Function Cleansing of Model 
def preprocess_text_model(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus URL dan tautan
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
    text = re.sub(r'pic.twitter.com.[\w]+', ' ', text)
    # Menghapus karakter yang tidak diinginkan, termasuk angka
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Menghapus kata 'user'
    text = text.replace('user', '')
    # Menghapus spasi berlebih
    text = re.sub(' +', ' ', text)
    # Menghapus karakter \n (newline)
    text = text.replace('\n', ' ')
    # menghapus kata 'url' 
    text = re.sub('url',' ', text)
    return text

def normalize_text_model(text):
    data_alay = pd.read_sql_query('select * from kamusalay', connection_model)
    dict_alay = dict(zip(data_alay['alay'], data_alay['normal'])) #Membungkus data teks_alay dan teks baku menjadi dictionary
    text_list = text.split()
    
    text_normal_list = [dict_alay.get(word, word) for word in text_list] #Mengambil nilai baku pada data teks_baku
    
    text_normal = ' '.join(text_normal_list) #mengganti teks yang tidak baku menjadi baku
    return text_normal.strip()
def stopword_removal(text):
    t_stopword = pd.read_sql_query('select * from stopword', connection_model)
    stopword_words = set(t_stopword['STOPWORD'])
    words = text.split()
    filtered_words = [word for word in words if word not in stopword_words]
    return ' '.join(filtered_words)

#==========================

#Function of Sensoring and Cleansing data without Modelling
def sensoring_text(str):
    df_abusive = pd.read_sql_query("select * from ABUSIVE", connection_data)
    dict_abusive = dict(zip(df_abusive['teks'], df_abusive['teks']))
    for x in dict_abusive:
        str = str.replace(x, '*' * len(x))  
    return str

def preprocessing_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = text.replace('user', '')
    words = [word for word in text.split() if len(word) > 2]
    return ' '.join(words)

def normalize_text(text):
    data_alay = pd.read_sql_query('select * from ALAY', connection_data)
    dict_alay = dict(zip(data_alay['teks_alay'], data_alay['teks_baku'])) #Membungkus data teks_alay dan teks baku menjadi dictionary
    text_list = text.split()
    
    text_normal_list = [dict_alay.get(word, word) for word in text_list] #Mengambil nilai baku pada data teks_baku
    
    text_normal = ' '.join(text_normal_list) #mengganti teks yang tidak baku menjadi baku
    return text_normal.strip()

def normalization_abusive(teks):
    df_abusive = pd.read_sql_query("select * from ABUSIVE", connection_data)
    dict_abusive = dict(zip(df_abusive['teks'],df_abusive['teks']))
    teks = teks.split()
    teks_normal = ''
    for str in teks:
        if(bool(str in dict_abusive)):
            str = sensoring_text(str)
            teks_normal = teks_normal + ' ' + str
        else:
            teks_normal = teks_normal + ' ' + str  
    teks_normal = teks_normal.strip()
    return teks_normal


def cleansing_sensoring (text):
    text = preprocessing_text(text)
    text = normalize_text (text)
    text = normalization_abusive(text)
    
    return text

def cleansing_model(text):
    string = preprocess_text_model(text)
    string = normalize_text_model(text)

    return string

# #Load LSTM_File
# lstm = open ("x_pad_sequences.pickle",'rb')
# lstm_file = pickle.load(lstm)
# lstm.close()

# #Model file
# lstm_model = load_model("main_model_new1.h5")


#Load Model NN
count_vect = pickle.load(open('NN_Files/feature_TFIDF_stopword.pickle', 'rb'))
model_NN = pickle.load(open('NN_Files/model_NN_TFIDF_stopword_Dataset.pickle', 'rb'))


@app.route('/')
def welcoming ():
    return 'Welcome to the API for cleansing and detecting sentiment text'
    

#Endpoint text sensoring
@swag_from('docs/text_implementing_processing.yml', methods=['POST'])
@app.route('/text_implementing_processing', methods=['POST'])
def text_implementing ():
    inputing_text = request.form.get('text')
    outputing_text = cleansing_sensoring(inputing_text)

    json_respon = {
        'input' : inputing_text,
        'output' :outputing_text
    }

    result_response = jsonify(json_respon)

    return result_response

#Endpoint file sensoring
@swag_from ('docs/file_uploading_sensoring.yml', methods=['POST'])
@app.route ('/file_uploading_sensoring', methods=['POST'])
def uploading_file():
    global connection_data  

    files = request.files['file']
    if not files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        data_csv = pd.read_csv(files, encoding="latin-1")
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Empty CSV file'}), 400

    data_csv = data_csv['Tweet']
    data_csv = data_csv.drop_duplicates()
    
    data_csv = data_csv.values.tolist()
    file_data = {}
    x = 0

    for string in data_csv:
        file_data[x] = {}
        file_data[x]['tweet'] = string
        file_data[x]['new_tweet'] = cleansing_sensoring(string)
        x += 1
    # Membuat DataFrame dari hasil cleaning data
    result_df = pd.DataFrame(file_data).T

    # M
    result_csv = result_df.to_csv(index=False)

    response = Response(
        result_csv,
        content_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=result.csv'
        }
    )

    return response

# #LSTM Teks Endpoint
# @swag_from('docs/LSTM_text.yml',methods=['POST'])
# @app.route('/LSTM_text',methods=['POST'])
# def text_lstm ():
#     text_input = request.form.get('text')
#     cleaned_text = [cleansing_model(text_input)]

    
#     feature = tokenizer.texts_to_sequences(cleaned_text)
#     feature_pad_sequences = pad_sequences(feature, maxlen=lstm_file.shape[1])

    
#     predict = lstm_model.predict(feature_pad_sequences)
#     polarity = np.argmax(predict[0])
#     sentiment_result = sentiment[polarity]

#     json_response = {
#         'status_code': 200,
#         'description': 'Hasil Prediksi LSTM Sentimen',
#         'data': {
#             'text': text_input,
#             'sentimen': sentiment_result
#         }
#     }
#     response_data = jsonify(json_response)
#     return response_data

# #LSTM File Endpoint
# @swag_from('docs/LSTM_file.yml',methods=['POST'])
# @app.route('/LSTM_file',methods=['POST'])
# def file_lstm():
#     file = request.files["Upload File"]
#     df = pd.read_csv(file, encoding="latin-1")
#     df = df.rename(columns={df.columns[0] :'text'})
#     df['data_bersih'] = df.apply(lambda rows : cleansing_model(rows['text']), axis =1)

#     result = []

#     for index, row in df.iterrows():
#         text = tokenizer.texts_to_sequences([(row['data_bersih'])])
#         feature_pad_sequences = pad_sequences(text, maxlen=lstm_file.shape[1])
#         predict = lstm_model.predict(feature_pad_sequences)
#         polarity = np.argmax(predict[0])
#         sentiment_result = sentiment[polarity]
#         result.append(sentiment_result)

#         original_text_upload = df.data_bersih.to_list()

#         json_response = {
#             'status_code' : 200,
#             'description' : 'Hasil Prediksi LSTM Sentimen',
#             'data' : {
#                 'tulisan' : original_text_upload,
#                 'sentimen' : result
#             }
#         }
#         response_data = jsonify(json_response)
#         return response_data
    
#NN Teks Endpoint
@swag_from('docs/NN_text.yml',methods=['POST'])
@app.route('/NN_text',methods=['POST'])
def text_NN ():
    text_input = request.form.get('text')
    cleaned_text = [cleansing_model(text_input)]

    text_feture_extration = count_vect.transform(cleaned_text)
    predict = model_NN.predict(text_feture_extration)
    polarity = np.argmax(predict[0])
    sentiment_result = sentiment[polarity]

    json_response = {
        'status_code': 200,
        'description': 'Hasil Prediksi NN Sentimen',
        'data': {
            'text': text_input,
            'sentimen': sentiment_result
        }
    }
    response_data = jsonify(json_response)
    return response_data

#NN File Endpoint
@swag_from('docs/NN_file.yml',methods=['POST'])
@app.route('/NN_file',methods=['POST'])
def file_NN():
    file = request.files["Upload File"]
    df = pd.read_csv(file, encoding="latin-1")
    df = df.rename(columns={df.columns[0] :'text'})
    df['data_bersih'] = df.apply(lambda rows : cleansing_model(rows['text']), axis =1)

    result = []

    for index, row in df.iterrows():
        text_feture_extration = count_vect.transform([(row['data_bersih'])])
        predict = model_NN.predict(text_feture_extration)
        polarity = np.argmax(predict[0])
        sentiment_result = sentiment[polarity]
        result.append(sentiment_result)
    original_text_upload = df.data_bersih.to_list()

    json_response = {
        'status_code' : 200,
        'description' : 'Hasil Prediksi NN Sentimen',
        'data' : {
            'tulisan' : original_text_upload,
            'sentimen' : result
        }
    }
    response_data = jsonify(json_response)
    return response_data



if __name__ == '__main__':
    app.run(debug=True)