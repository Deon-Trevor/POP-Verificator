from matplotlib import pyplot as plt
from pytesseract import Output
import pytesseract
import cv2
import os
from collections import namedtuple
from PIL import Image
from colorama import Fore
from pdf2image import convert_from_path
import argparse
import os
import joblib
import imutils
import regex as re
import subprocess 
from subprocess import check_output
import json
import datefinder
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nameparser.parser import HumanName
from nltk.corpus import wordnet
from dateparser.search import search_dates
from datetime import datetime
from flask import Flask,jsonify
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import io
import numpy as np

# Your API definition
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

  
# def pdf_to_images_to_text():
#     """
#     Input: Pdf path
#     Output: First page converted to image
#     """
#     images = img
#     text=""

#     for i in range(len(images)):
#     # Save pages as images in the pdf
#         images[i].save('new-image'+ str(i) +'.jpg', 'JPEG')       

        
#         text += pytesseract.image_to_string(images[i])
            
#     print(text, file=open("clientdata.txt", "w"))

#     classify_document(text)
    
       

def classify_document(text):
    # load model
    model = joblib.load("text_model.pkl")
    text=[text]
    # use model to predict classification for query
    classification_labels = model.predict(text)
    
    print(classification_labels)

    

@app.route('/', methods=['POST'])
    
def extract_data():
    """
    Input: Text from image
    Output: Client Data
    """

    photo = request.files['photo']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)

    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
    rot_data = pytesseract.image_to_osd(rgb)
  
    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)

    angle = float(rot)

    # rotate the image to deskew it
    rotated = imutils.rotate_bound(img, angle) #added
    
    
    # Run tesseract OCR on image
    text = pytesseract.image_to_string(rotated, lang='eng', config="--psm 6") 
     
    
    print(text, file=open("clientdata.txt", "w"))
    classify_document(text)




    #Name of client
    os.system("cat clientdata.txt | grep -i 'made by' >> proof_of_payment.txt")
    os.system("cat clientdata.txt | grep -i 'Name' >> proof_of_payment.txt")
    os.system("cat clientdata.txt | grep -i 'paid' >> proof_of_payment.txt")
    
    #Date of transaction
    os.system("cat clientdata.txt | grep -i 'Date' >> proof_of_payment.txt")
    os.system("cat clientdata.txt | grep -oE '[0-9]{2}/[0-9]{2}/[0-9]{4}' >> proof_of_payment.txt")
    os.system("cat clientdata.txt | grep -oE '[0-9]{4}/[0-9]{2}/[0-9]{2}' >> proof_of_payment.txt")

    #Amount in ZAR
    os.system("cat clientdata.txt | grep -oE '\R[0-9]+(.[0-9]+)?' >> proof_of_payment.txt")
    os.system("cat clientdata.txt | grep -i 'deposit' >> proof_of_payment.txt")
    os.system("cat clientdata.txt | grep -i 'amount' >> proof_of_payment.txt")
    

    if os.stat("proof_of_payment.txt").st_size == 0:
        print(Fore.RED + "\nNot Valid Proof Of Payment\n")
        outcome=False
        valid_status={'isValid':outcome, 'name': 'Null', 'paidAt': 'Null', 'amount': 'Null'} 
        
        
    else:
        print("\nValid Proof Of Payment\n")
        os.system("cat proof_of_payment.txt")
       
        outcome=True
        #deal with data
        with open("proof_of_payment.txt",'r') as file:
            String=file.read()
            
        extracted_dates = []

        # Returns a list of tuples of (substring containing the date, datetime.datetime object)
        dates = search_dates(String)


        if dates is not None:
            for d in dates:
                extracted_dates.append(str(d[1]))
        else:
            extracted_dates.append('None')

        String = re.sub('[/|.|@|%|\d+]', '', String)
        Sentences = nltk.sent_tokenize(String)
        Tokens = []
        for Sent in Sentences:
            Tokens.append(nltk.word_tokenize(Sent)) 
        Words_List = [nltk.pos_tag(Token) for Token in Tokens]

        Nouns_List = []

        for List in Words_List:
            for Word in List:
                if re.match('[NN.*]', Word[1]):
                    Nouns_List.append(Word[0])

        Names = []
        for Nouns in Nouns_List:
            if not wordnet.synsets(Nouns):
                Names.append(Nouns)
        file.close()
        
        with open ("proof_of_payment.txt",'r') as dot:
            lines=dot.read()
        
            #pattern= r"(?:[\R](?<!\d)(?:\d{1,3}(?:,\d{3})*|\d{4,})?\.?\d+)"
            pattern=r"(?:\b(?:[BS]/\.|R(?:D?\$|p))|\b(?:[TN]T|[CJZ])\$|Дин\.|\b(?:Bs|Ft|Gs|K[Mč]|Lek|B[Zr]|k[nr]|[PQLSR]|лв|ден|RM|MT|lei|zł|USD|GBP|EUR|JPY|CHF|SEK|DKK|NOK|SGD|HKD|AUD|TWD|NZD|CNY|KRW|INR|CAD|VEF|EGP|THB|IDR|PKR|MYR|PHP|MXN|VND|CZK|HUF|PLN|TRY|ZAR|ILS|ARS|CLP|BRL|RUB|QAR|AED|COP|PEN|CNH|KWD|SAR)|\$[Ub]|[\p{Sc}ƒ])\s?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?(?!\.?\d)"
            
            match=re.findall(pattern,lines)
            amount=" ".join(str(x) for x in match)

        
            valid_status={'isValid':outcome, 'name': Names, 'paidAt': extracted_dates[0], 'amount': amount} 
        dot.close()
    os.system("rm proof_of_payment.txt") 
    
    return jsonify(valid_status)
   
     


if __name__ == '__main__':
    
    app.run(debug=True, port=4455)