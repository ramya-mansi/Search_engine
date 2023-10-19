from PyPDF2 import PdfReader
import re
import pandas as pd

df=pd.read_csv('info_new.csv')

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')

l=[]
id=0
p=[]

for i in range(1,61):
    try:
        reader = PdfReader(f"paper_{i}.pdf")
        for j in range(len(reader.pages)):
            page = reader.pages[j]
            original_string = page.extract_text().lower()
            modified_string = original_string.replace('\n', ' ')
            value = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", modified_string)
            value = re.sub(r'\d+', ' ', value)
            result = re.sub(r'\b[a-zA-Z]\b', '', value)
            result = re.sub(r'\s+', ' ', result)
            v = " ".join([word for word in result.split() if word not in (stop)])
            v = re.sub(r'\b(?!(ai)\b)[a-zA-Z]{2}\b', '', v)
            index = v.find('abstract')
            if index != -1:
                v = v[index + len('abstract'):].strip()
            index = v.find('references')
            if index != -1:
                v = v[:index].strip()
            p.append(j+1)
            l.append(v)
        print(f"data cleaned for paper_{i}.pdf")          
    except:
        continue
    
df['Cleaned Data'] = l 

df.to_csv('thurs.csv')

print(df['Cleaned Data'][4])