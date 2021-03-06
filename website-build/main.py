from flask import Flask, render_template, request, Markup

import predictions
import pandas as pd
import base64
import matplotlib.pyplot as plt
import plotly.express as px
#import chart_studio.tools as tls

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

app= Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
@app.route("/index.html", methods=['POST', 'GET'])
def index():
    #blank defaults so it runs on startup
    main_text1 = 'In the box below, input your Twitter username (without the @ symbol), and the model will predict the most likely fiction genre of you writing based on your 100 most recent tweets. '
    main_text2 = 'Note: The process may take a moment and will refresh the page. Make sure to scroll for results'
    text_scroll=''
    text_in = ''
    df = pd.DataFrame()
    chart = ''
    error_text=''
    results = 'Twitter Test'
    in_text = ""
    word_title = ''
    word_text = ["",""]
    prob_text = ''
    tweet_test = '<blockquote class="twitter-tweet" data-conversation="none"><p lang="en" dir="ltr"><a href="https://twitter.com/jimmyfallon/status/1245871404110606337"></a></blockquote><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>'
    html_strings = []
    initial_twitter_text = ["","",""]
    hrs = ''
    x_data = []
    y_data = []
    top_text = ["","","","",""]
    graph_js = ""
    if request.method == 'POST':
        text_scroll='Scroll for Results!'
        text_in = request.form['text_in']
        try:
            df, wordcloud, top_tweets, html_strings,best_genre = predictions.final_tweet_model(text_in)
            results = 'Results'
            in_text = "Input text: "
            main_text1 = 'Try another username'
            main_text2 = ''
            prob_text = 'Graph of Probability of Genres'
            wordcloud.to_file("/home/sgunners/main/wc.png")
            with open("/home/sgunners/main/wc.png","rb") as img_file:
                my_string = base64.b64encode(img_file.read()).decode()
            chart = Markup('<img src="data:image/png;base64,{}" width: 600px; height: 400px>'.format(my_string))
            word_title = "Word Cloud"
            word_text = ["Below is a word cloud of the words that indicated the most probable genre of the input, with larger words having a larger effect on the model. The word cloud is generated by taking the top genre and applying the best genre model to those words. The size of the words are then determined by how large of an effect the word has on predicting the top genre. ", "The below word cloud is for {}".format(best_genre)]
            initial_twitter_text = ["Your most likely genre is: ",best_genre,"Below is the tweet that fits {} best".format(best_genre),"It fits at a predicted value of ", round(top_tweets['pred_prob'][0],5)*100]
            hrs='<hr />'
            df=df.head(5)
            df = df.sort_values(by='Probability', ascending=True)
            x_data = list(df['Probability'])
            y_data = list(df['index'])
            top_text = predictions.result_text(best_genre,text_in)
            graph_js = '<div id="plotly-div"></div>'
        except:
            error_text='Invalid Username. Check spelling and try again'
            in_text = "Username Entered: "
    return render_template("index.html",
                            text_scroll=text_scroll,
                            text_in=text_in,
                            title='Results',
                            chart = chart,
                            error_text=error_text,
                            results = results,
                            in_text = in_text,
                            word_title = word_title,
                            word_text = word_text,
                            prob_text=prob_text,
                            main_text1=main_text1,
                            main_text2=main_text2,
                            tweet_test=tweet_test,
                            html_strings = html_strings,
                            initial_twitter_text=initial_twitter_text,
                            hrs=hrs,
                            x_data=x_data,
                            y_data=y_data,
                            top_text=top_text,
                            graph_js=graph_js)

@app.route("/summ.html", methods=['POST','GET'])
def summ():
   #blank defaults so it runs on startup
    text_scroll=''
    text_in = ''
    text_out = ''
    df = pd.DataFrame()
    prob_text = ''
    bar=''
    chart = ''
    error_text=''
    results = "Genre Discovery - Summaries"
    in_text = ""
    word_title = ''
    word_text = ["",""]
    fic_nonfic_text=''
    hrs=""
    x_data = ""
    y_data = ""
    graph_js = ""
    if request.method == 'POST':
            text_scroll='Scroll for Results!'
            text_in = request.form['text_in']
            value, df, wordcloud, best_genre= predictions.final_text_model(text_in)
            fic_nonfic_text = "The probability of your text being non-fiction is {:.2%} and the probability of your text being fiction is {:.2%}".format(value,1-value)
            results = 'Results'
            in_text = "Input text: "
            hrs='<hr />'
            if value<0.5:
        	    text_out = "This is Fiction"
        	    prob_text = 'Graph of Probability of Genres'
        	    df = df.sort_values('Probability')
        	    wordcloud.to_file("/home/sgunners/main/wc_summ.png")
        	    with open("/home/sgunners/main/wc_summ.png","rb") as img_file:
        	        my_string = base64.b64encode(img_file.read()).decode()
        	    chart = Markup('<img src="data:image/png;base64,{}" width: 600px; height: 400px>'.format(my_string))
        	    word_title = "Word Cloud"
        	    word_text = ["Below is a word cloud of the words that indicated the most probable genre of the input, with larger words having a larger effect on the model. The word cloud is generated by taking the top genre and applying the best genre model to those words. The size of the words are then determined by how large of an effect the word has on predicting the top genre.", "The below word cloud is for {}".format(best_genre)]
        	    df = df.head(5)
        	    df = df.sort_values(by='Probability', ascending=True)
        	    x_data = list(df['Probability'])
        	    y_data = list(df['index'])
        	    graph_js = '<div id="plotly-div"></div>'
            else:
        	    text_out = "This is Non-Fiction"
    return render_template("summ.html",
                            text_scroll=text_scroll,
                            text_in=text_in,
                            text_out=text_out,
                            title='Results',
                            chart = chart,
                            error_text=error_text,
                            results = results,
                            in_text = in_text,
                            word_title = word_title,
                            word_text = word_text,
                            bar=bar,
                            prob_text=prob_text,
                            fic_nonfic_text=fic_nonfic_text,
                            hrs=hrs,
                            x_data=x_data,
                            y_data=y_data,
                            graph_js = graph_js)

@app.route("/project_write.html")
def project_write():
    return render_template("project_write.html")

@app.route("/barack_obama.html")
def bo():
    df, wordcloud, top_tweets, html_strings,best_genre = predictions.final_tweet_model('BarackObama')
    initial_twitter_text = ["Your most likely genre is: ",best_genre,"Below is the tweet that fits {} best".format(best_genre),"It fits at a predicted value of ", round(top_tweets['pred_prob'][0],5)*100]
    hrs='<hr />'
    word_title = "Word Cloud"
    word_text = ["Below is a word cloud of the words that indicated the most probable genre of the input, with larger words having a larger effect on the model. The word cloud is generated by taking the top genre and applying the best genre model to those words. The size of the words are then determined by how large of an effect the word has on predicting the top genre. ", "The below word cloud is for {}".format(best_genre)]
    wordcloud.to_file("/home/sgunners/main/wc_bo.png")
    with open("/home/sgunners/main/wc_bo.png","rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode()
    chart = Markup('<img src="data:image/png;base64,{}" width: 600px; height: 400px>'.format(my_string))
    df=df.head(5)
    df = df.sort_values(by='Probability', ascending=True)
    x_data = list(df['Probability'])
    y_data = list(df['index'])
    top_text = predictions.result_text(best_genre,'BarackObama')
    graph_js = '<div id="plotly-div"></div>'
    return render_template("barack_obama.html",initial_twitter_text=initial_twitter_text, html_strings=html_strings, hrs=hrs, word_title=word_title, word_text=word_text, chart=chart, x_data=x_data, y_data=y_data, top_text=top_text, graph_js=graph_js)

@app.route("/goodcaptain.html")
def good():
    df, wordcloud, top_tweets, html_strings,best_genre = predictions.final_tweet_model('goodcaptain')
    initial_twitter_text = ["Your most likely genre is: ",best_genre,"Below is the tweet that fits {} best".format(best_genre),"It fits at a predicted value of ", round(top_tweets['pred_prob'][0],5)*100]
    hrs='<hr />'
    word_title = "Word Cloud"
    word_text = ["Below is a word cloud of the words that indicated the most probable genre of the input, with larger words having a larger effect on the model. The word cloud is generated by taking the top genre and applying the best genre model to those words. The size of the words are then determined by how large of an effect the word has on predicting the top genre. ", "The below word cloud is for {}".format(best_genre)]
    wordcloud.to_file("/home/sgunners/main/wc_gc.png")
    with open("/home/sgunners/main/wc_gc.png","rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode()
    chart = Markup('<img src="data:image/png;base64,{}" width: 600px; height: 400px>'.format(my_string))
    df=df.head(5)
    df = df.sort_values(by='Probability', ascending=True)
    x_data = list(df['Probability'])
    y_data = list(df['index'])
    top_text = predictions.result_text(best_genre,'goodcaptain')
    graph_js = '<div id="plotly-div"></div>'
    return render_template("goodcaptain.html",initial_twitter_text=initial_twitter_text, html_strings=html_strings, hrs=hrs, word_title=word_title, word_text=word_text, chart=chart, x_data=x_data, y_data=y_data, top_text=top_text, graph_js=graph_js)


@app.route("/mayoremanuel.html")
def mayor():
    df, wordcloud, top_tweets, html_strings,best_genre = predictions.final_tweet_model('mayoremanuel')
    initial_twitter_text = ["Your most likely genre is: ",best_genre,"Below is the tweet that fits {} best".format(best_genre),"It fits at a predicted value of ", round(top_tweets['pred_prob'][0],5)*100]
    hrs='<hr />'
    word_title = "Word Cloud"
    word_text = ["Below is a word cloud of the words that indicated the most probable genre of the input, with larger words having a larger effect on the model. The word cloud is generated by taking the top genre and applying the best genre model to those words. The size of the words are then determined by how large of an effect the word has on predicting the top genre. ", "The below word cloud is for {}".format(best_genre)]
    wordcloud.to_file("/home/sgunners/main/wc_me.png")
    with open("/home/sgunners/main/wc_me.png","rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode()
    chart = Markup('<img src="data:image/png;base64,{}" width: 600px; height: 400px>'.format(my_string))
    df=df.head(5)
    df = df.sort_values(by='Probability', ascending=True)
    x_data = list(df['Probability'])
    y_data = list(df['index'])
    top_text = predictions.result_text(best_genre,'mayoremanuel')
    graph_js = '<div id="plotly-div"></div>'
    return render_template("mayoremanuel.html",initial_twitter_text=initial_twitter_text, html_strings=html_strings, hrs=hrs, word_title=word_title, word_text=word_text, chart=chart, x_data=x_data, y_data=y_data, top_text=top_text, graph_js=graph_js)


