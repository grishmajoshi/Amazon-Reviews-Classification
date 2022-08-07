import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


names = ['Logistic', 'Multinomial','Bernoulli']

models = {i:pickle.load(open(f'{i}.pkl', 'rb')) for i in names}

def get_sentiment(model, testData):
    countVector = pickle.load(open("countVector.pkl",'rb'))
    tfidf_transformer =  pickle.load(open("tfidf_transformer.pkl",'rb'))
    testCounts = countVector.transform([testData])
    testTfidf = tfidf_transformer.transform(testCounts)
    for mod, clf in model.items():
        result = clf.predict(testTfidf)[0]
        probability = clf.predict_proba(testTfidf)[0]
        st.write(f"Text Classified by {mod} Model is {result.upper()}: negative prob {round(probability[0],4)}, positive prob {round(probability[1],4)}")
      

# st.set_page_config(layout="wide")
st.title("AMAZON FOOD REVIEW - SENTIMENT ANALYSIS")
text = st.text_input("Enter the text ")
submit = st.button("SUBMIT")
if submit:
    get_sentiment(models,text)
    
