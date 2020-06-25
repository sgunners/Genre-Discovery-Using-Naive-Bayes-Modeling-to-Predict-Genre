#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import tweepy as tw
nltk.download('punkt')
nltk.download('stopwords')
import joblib
from wordcloud import WordCloud
from scipy.sparse import diags
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import plotly.express as px
import chart_studio.plotly as py

cmu_summ_full = joblib.load('/home/sgunners/main/nb_builds.p')

def clean_text(input_text):
    new_clean = input_text.replace('\n',' ').replace('\r', '')
    new_clean = new_clean.replace('"','')
    return new_clean

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J":wordnet.ADJ,
               "N":wordnet.NOUN,
               "V":wordnet.VERB,
               "R":wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemma_all(doc):
    lemmatizer=WordNetLemmatizer()
    var = [lemmatizer.lemmatize(w, get_wordnet_pos((w))) for w in nltk.word_tokenize(doc)]
    return ' '.join(var)

def check_ficnonfic(test_text):
    test_text = cmu_summ_full['FicnonFic']['count_vect'].transform(test_text)
    prob = cmu_summ_full['FicnonFic']['count_model'].predict_proba(test_text)[0][0]
    return prob

def check_genre(test_text,genre_dict, vect,model):
    test_text = genre_dict['count_vect'].transform(test_text)
    prob = genre_dict['count_model'].predict_proba(test_text)[0][0]
    return prob

def genre_plot(df):
    df = df.sort_values(by="Probability", ascending=True)
    fig = px.bar(df, x="Probability", y="index", orientation="h")
    fig.update_layout(title={'text':'Most Probable Genres','y':0.95,'x':0.5,'xanchor':'center','yanchor':'top'},
                     yaxis_title="")
    fig.update_traces(marker_color='rgb(129,92,146)', marker_line_color='rgb(14,17,22)')
    url = py.plot(fig, filename='current',auto_open=False)
    return url

def test_genre(test_test):
    genre_dict_list = [cmu_summ_full['Speculative fiction'],cmu_summ_full['Science fiction'],cmu_summ_full['Fantasy'],
                       cmu_summ_full["Children's literature"], cmu_summ_full['Mystery'],cmu_summ_full['Historical fiction'],
                       cmu_summ_full['Young adult literature'],cmu_summ_full['Suspense'], cmu_summ_full['Horror'],
                       cmu_summ_full['Thriller']]
    genre_list = ['Speculative fiction','Science fiction','Fantasy',"Children's literature",'Mystery','Historical fiction',
                 'Young adult literature','Suspense','Horror','Thriller']
    genre_vect_list = ['tfidf_vect','count_vect','count_vect','count_vect','tfidf_vect','count_vect','count_vect','count_vect',
                      'tfidf_vect','tfidf_vect','count_vect']
    genre_model_list = ['tfidf_model','count_model','count_model','count_model','tfidf_model','count_model','count_model',
                       'count_model','tfidf_model','tfidf_model','count_model']
    genre_values = {}
    genre_dict_full = {}
    i=0
    for genre_dict in genre_dict_list:
        genre_values[genre_list[i]] = check_genre(test_test,genre_dict, genre_vect_list[i],genre_model_list[i])
        genre_dict_full[genre_list[i]]={'vect':genre_vect_list[i], 'model':genre_model_list[i]}
        i = i+1
    genre_values = pd.DataFrame.from_dict(genre_values, orient='index')
    genre_values.columns = ['Probability']
    genre_values = genre_values.sort_values(by='Probability', ascending=False)
    best_genre = genre_values.index.values[0]
    text = cmu_summ_full[best_genre][genre_dict_full[best_genre]['vect']].transform(test_test)
    text = np.transpose(text)
    text = text.toarray()
    text = text.transpose()
    text = diags(text,[0])
    wc_pred = cmu_summ_full[best_genre][genre_dict_full[best_genre]['model']].predict_proba(text)[:,0]
    wc_pred = wc_pred*text
    vocab = cmu_summ_full[best_genre][genre_dict_full[best_genre]['vect']].get_feature_names()
    wc_build = dict(zip(vocab,wc_pred))
    wordcloud = WordCloud().generate_from_frequencies(wc_build)
    return genre_values, wordcloud, genre_dict_full[best_genre],best_genre

def final_text_model(test_text):
    text = clean_text(test_text)
    text = lemma_all(text)
    text = [text]
    value = check_ficnonfic(text)
    df = pd.DataFrame()
    wordcloud = WordCloud()
    if value<0.5:
        df, wordcloud,best_genre_dict, best_genre = test_genre(text)
        df['Probability']=round(df['Probability']/sum(df['Probability']),5)*100
        df = df.sort_values('Probability', ascending=False).reset_index()
    return value, df, wordcloud, best_genre


#twitter keys (ADD correct values here as needed!)
consumer_key = 
consumer_secret = 
access_token = 
access_token_secret = 


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth,parser=tw.parsers.JSONParser())


def clean_tweets(username):
    alltweets = []
    new_tweets = api.user_timeline(screen_name = username, count=100, include_rts=False)
    alltweets.extend(new_tweets)
    outtweets = [[tweet['text'],tweet['id']] for tweet in alltweets]
    fo = pd.DataFrame(outtweets)
    outtweets = pd.DataFrame(fo[0])

    text_all = ''
    for tweet in outtweets[0]:
        tweet_blob = TextBlob(tweet)
        text_all = text_all + ' '.join(tweet_blob.words)
    tweet_list = [ele for ele in text_all.split() if ele !='user']
    clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word.lower() for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    check_tweets = outtweets
    check_tweets['words']=check_tweets[0].map(lambda x: ' '.join(TextBlob(x).words))
    check_tweets['token']=check_tweets['words'].map(lambda x: [ele for ele in x.split() if ele !='user'])
    check_tweets['c_tokens']= check_tweets['token'].map(lambda x: [t for t in x if re.match(r'[^\W\d]*$', t)])
    check_tweets['c_tokens']= check_tweets['c_tokens'].map(lambda x:[word.lower() for word in x if word.lower() not in stopwords.words('english')])
    check_tweets['c_tokens']= check_tweets['c_tokens'].map(lambda x: ' '.join(x))
    return ' '.join(clean_mess), check_tweets, fo

def url_form(tweet_id,username):
    url = "https://twitter.com/{}/status/{}".format(username,tweet_id)
    html = '<blockquote class="twitter-tweet" data-conversation="none"><p lang="en" dir="ltr"><a href="{}"></a></blockquote><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>'.format(url)
    return html

def final_tweet_model(test_text):
    text, outtweets, fo = clean_tweets(test_text)
    text = lemma_all(text)
    text = [text]
    df = pd.DataFrame()
    wordcloud = WordCloud()
    best_genre_dict={}
    df, wordcloud,best_genre_dict, best_genre = test_genre(text)
    df['Probability']=round(df['Probability']/sum(df['Probability']),5)*100
    df_graph = df
    best_tweet=''
    test_tweets = cmu_summ_full[best_genre][best_genre_dict['vect']].transform(outtweets['c_tokens'])
    test_proba = cmu_summ_full[best_genre][best_genre_dict['model']].predict_proba(test_tweets)
    best_value = max(list(next(zip(*test_proba))))
    outtweets['pred_prob']=pd.Series(list(next(zip(*test_proba))))
    best_tweet = outtweets[outtweets['pred_prob']==best_value].reset_index()
    final_best = fo[fo[0]==best_tweet[0][0]].reset_index()
    fo['pred_prob']=pd.Series(list(next(zip(*test_proba))))
    fo = fo.sort_values('pred_prob', ascending=False).reset_index().drop('index', axis=1)
    html_strings = fo[1][0:3].apply(lambda x: url_form(x, test_text))
    df = df.sort_values('Probability', ascending=False).reset_index()
    return df, wordcloud, fo[0:3], html_strings,best_genre

def result_text(best_genre, text_in):
    full_html = ['<div style="width:500px;height:90px;border: 2px solid #815c92">', '<b><a href="https://twitter.com/{}">@{}</a></b>'.format(text_in, text_in), "<br>", "<b>Your most likely genre is: {}</b>".format(best_genre), '</div>']
    return full_html

