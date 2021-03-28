#importing libraries
from flask import Flask,url_for,render_template,request
from flaskext.markdown import Markdown
# NLP Pkgs
import spacy
from spacy import displacy
nlp = spacy.load('en')
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
# Init
app = Flask(__name__)
Markdown(app)
run_with_ngrok(app)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/extract',methods=["GET","POST"])
def extract():
	if request.method == 'POST':
		raw_text = request.form['rawtext']
		docx = nlp(raw_text)
		html = displacy.render(docx,style="ent")
		html = html.replace("\n\n","\n")
		result = HTML_WRAPPER.format(html)

	return render_template('result.html',rawtext=raw_text,result=result)


@app.route('/previewer')
def previewer():
	return render_template('previewer.html')

@app.route('/preview',methods=["GET","POST"])
def preview():
	if request.method == 'POST':
		newtext = request.form['newtext']
		result = newtext

	return render_template('preview.html',newtext=newtext,result=result)

@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == '__main__':
    import portpicker
    port = portpicker.pick_unused_port()
    from google.colab import output
    output.serve_kernel_port_as_window(port)
    from gevent.pywsgi import WSGIServer
    host='localhost'
    app_server = WSGIServer((host, port), app)
    app_server.serve_forever()