import streamlit as st
import altair as alt
import plotly.express as px

import transformers
from transformers import pipeline
analyzer = pipeline("sentiment-analysis")

import pandas as pd
import numpy as np
from datetime import datetime

import joblib

pipe_lr=joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))
#pipe_eimg=joblib.load(open("models/emotions_vgg19.pkl","rb"))
#pipe_simg=joblib.load(open("models/sentiment_vgg19.pkl","rb", encoding="UTF-8"))

from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details2,view_all_prediction_details,create_emotionclf_table3


import snscrape.modules.twitter as sntwitter
import pandas as pd


st.markdown("""
<link rel="shortcut icon" href="favicon.jpg"/>
<title>Emotion Detection System</title>
<style>
.css-9s5bis.edgvbvh3
{
    visibility: hidden;
}
.css-1q1n0ol.egzxvld0
{
   visibility: hidden;
}
</style>

""", unsafe_allow_html=True)




def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def predict_emotionsImg(fileImg):
    results = pipe_lr.predict(fileImg)
    return results[0]

def predict_sentiImg(docx):
    results = pipe_lr.predict(fileImg)
    return results[0]

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("Emotion Detector System")
    menu = ["Home","About","Detector","Tweets","History","Contact"]
    choice = st.sidebar.selectbox("Menu",menu)
    create_page_visited_table()
    create_emotionclf_table3()

    if choice == "Detector":
        add_page_visited_details("Detector",datetime.now())
        st.subheader("Emotion In Text")

        with st.form(key='emotion_clf_form'):
            #imgF=st.file_uploader("Upload a file")
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

            if submit_text:
                col1,col2  = st.columns(2)

                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                pred=analyzer(raw_text)


                with col1:
                    st.success("Original Text")
                    st.write(raw_text)
                    st.success("Prediction")
                    #st.write(pred)
                    st.write("Label: {}".format(pred[0]["label"]))
                    st.write("Score: {}".format(pred[0]["score"]))

                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction,emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    # st.write(probability)
                    global proba_df
                    proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                    proba_df[pred[0]["label"]]=pred[0]["score"]
                    
                    #global proba_df
                    # st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions","probability"]

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                    st.altair_chart(fig,use_container_width=True)
                    ll=pred[0]["label"]
                    sc=pred[0]["score"]
                    add_prediction_details2(raw_text,ll,sc,prediction,np.max(probability),datetime.now())

    elif choice=="Tweets":
        add_page_visited_details("Tweets",datetime.now())

        with st.form(key="tweet_collection_form"):
            twhan=st.text_input("Enter twitter handle: ")
            tl_f=st.number_input("Enter how many tweets you want to retrive:")
            fdt=st.date_input("Enter Date From")
            tdt=st.date_input("Enter Date To")
            submit_text2 = st.form_submit_button(label='Submit')
            
            if submit_text2:

                ################          

                #query="python"
                query="(from:"+twhan+") until:"+str(tdt)+" since:"+str(fdt)
                tweets=[]
                limits=tl_f

                for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                    #print(vars(tweet))
                    #break
                    if len(tweets)==limits:
                        break
                    else:
                        tweets.append([tweet.date, tweet.user.username, tweet.content])

                dft=pd.DataFrame(tweets, columns=['Date','user','Tweet'])
                #print(dft)


                #####
                st.dataframe(dft)
            else:
                ################
                st.write(fdt)
                st.write(tdt)          

                #query="python"
                query="(from:raseluddin102) until:2022-12-24 since:2006-01-01"
                tweets=[]
                limits=5000

                for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                    #print(vars(tweet))
                    #break
                    if len(tweets)==limits:
                        break
                    else:
                        tweets.append([tweet.date, tweet.user.username, tweet.content])

                dft=pd.DataFrame(tweets, columns=['Date','user','Tweet'])
                #print(dft)


                #####
                st.dataframe(dft)

    elif choice=="History":
        add_page_visited_details("History",datetime.now())
        st.subheader("History")
        st.text("Check previous detection history and also page view from here.")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
            st.dataframe(page_visited_details)
            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
            st.altair_chart(c,use_container_width=True)

            p = px.pie(pg_count,values='Counts',names='Pagename')
            st.plotly_chart(p,use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Predicted','max_score', 'Prediction','Probability','Time_of_Visit'])
            #df_emotions["Predicted"]=proba_df[pred[0]["label"]] #=pred[0]["score"]
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
            st.altair_chart(pc,use_container_width=True)
    elif choice=="Contact":
        add_page_visited_details("Contact",datetime.now())
        st.subheader("Contact Us")
        st.write("If you have any question or any query, feel free to knock us in the below address or email.")
        st.write("Email: raseluddin102@gmail.com")
        st.write("Baridhara, Dhaka-1212, Bangladesh.")
    elif choice=="About":
        add_page_visited_details("About",datetime.now())

        st.header("About this app")
        #st.header("Functions:")
        st.subheader("Emotion Detection From Text")
        st.write("People express their emotion by face or by speech.Speech converted to text have also indication of emotion.We can this detect this emotion. This system can detect this emotion nicely.")
        st.write("Motivation:")
        st.write("Twitter is one of the most popular social media in this era. People use this social media for various reaseon. They post their daily activity on twiitter. Young teenager use this social media more. They post on social media whether they they are happy or sad. If we can detect this emotion, we could save many suicide case.")
        st.write("Feature of this system:")
        st.write("1. This system can detect emotion from text.    2. This system can retrive twitter post from any twiiter user.    3. This system can retrive twitter post upto 5000.    4. Twiiter post can be collected from any date.    5. EDS can also predict the nearest emotion of the twiiter post.    6. This software can also preview the software usage history in nice graph.")
        st.subheader("Emotion Detection from face")
        st.write("In the next version, we will add this feature to this software. As said before, face have present emotion. Human can easily understand the emotion from face. But this app will also predict emotion from image also. Wait and try it out till then...")

    elif choice=="Home":
        add_page_visited_details("Home",datetime.now())

        #st.image("img//image.jpg")        
    
        #add_page_visited_details("About",datetime.now())
        st.image(["img//download.jpg", "img//1download.png", "img//2Capture.png"])
        #st.image("img//1download.png") 2Capture.JPG

        #st.write("")
        st.header("Detect emotion from text")
        st.subheader("How to Use")
        st.write("To detect emotion from twitter post, follow below...")
        st.write("At first collect twitter post of your desired person.")
        st.write("To collect twiitter post, go to Tweets page from navigation bar. In Tweets page, you have to provide twiiter handle, number of how many tweets you want to get, which date from you want to get tweet till which date in the form. Then you will submit. After That you will get your desired twitter post of your expected user. Then copy the tweets. By default you will get demo twi post of a user named raseluddin102.   ")
        st.write("Then...")
        st.write("Go to Detector page. Here you have to paste the copied tweets from Tweets page and submit it. You will see the prediction in the below as lebel and also in graph. CONGRATULATIONS, you have successfully predict the emotion from twiiter.")




if __name__ == '__main__':
	main()


            





























