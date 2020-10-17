
# Code is property of Vimanyu Devgan
#email : vimanyudevgan@gmail.com

# In[1]:


import pandas as pd
true_df = pd.read_csv('True.csv')# reading the the examples of real news
false_df = pd.read_csv('fake.csv')# reading the the examples of fake news
true_df['target'] = 1
false_df['target'] = 0


# In[2]:


final_df = pd.concat([true_df,false_df])


# In[3]:


final_df.sample(frac = 0.5)


# In[4]:


final_df['full_text'] = final_df['title'] +''+ final_df['text'] +''+ final_df['subject'] 


# In[5]:


final_data = pd.DataFrame()


# In[6]:


final_data = final_df[['full_text','target']]


# In[7]:


final_data.head()


# In[8]:


final_data['full_text']  = final_data['full_text'].apply(lambda x: x.lower())


# In[9]:


import string
def punc_remo(str):
    clean_str = ''
    for char in str:
        if(char not in string.punctuation):
            clean_str = clean_str + char
    return clean_str
    


# In[10]:


final_data['full_text']  = final_data['full_text'].apply(punc_remo)


# In[11]:


final_data['full_text']


# In[12]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[13]:


final_data['full_text'].apply(lambda x: [item for item in x if item not in stop])


# In[14]:


final_data['full_text']


# In[15]:


def stop_remo(str):
    clean_str = ''
    for char in str.split():
        if(char not in stop):
            clean_str = clean_str +' '+ char
    return clean_str


# In[16]:


final_data['full_text'] = final_data['full_text'].apply(stop_remo)


# In[17]:


final_data['full_text']


# In[18]:


print (stop)


# In[19]:


final_data['full_text'].apply(lambda x: [item for item in x.split() if item not in stop])


# In[20]:


final_data['full_text']


# In[21]:


import matplotlib.pyplot as plt
from nltk import tokenize
token = tokenize.WhitespaceTokenizer()
import seaborn as sb
import nltk
def pareto_graph(data, qty):
    s = data['full_text'].to_string()
    tokens = token.tokenize(s)
    freq = nltk.FreqDist(tokens)
    print(freq)
    temp_df = pd.DataFrame({"word":list(freq.keys()),"Frequency": list(freq.values())})
    print(temp_df)
    temp_df = temp_df.nlargest(columns = "Frequency", n = qty)
    print(temp_df)
    plt.figure(figsize =(15,12))
    plot = sb.barplot(data = temp_df, x = 'word', y = 'Frequency', color = 'green')
    plt.show()


# In[22]:


pareto_graph(final_data, 20)


# In[34]:


from nltk import PorterStemmer


# In[35]:


porter = PorterStemmer()


# In[36]:


print(porter.stem(final_data['full_text']))


# In[37]:


def stem(data):
    stemmed = ''
    for word in data.split():
        stemmed = stemmed + ' ' + porter.stem(word)
    return stemmed


# In[ ]:


final_data["full_text"]= final_data['full_text'].apply(stem)


# In[51]:


final_data["full_text"]


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer().fit(final_data['full_text'])
print(bow)
bow_final = bow.transform(final_data['full_text'])
print(bow_final)



# In[41]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit(bow_final)
tfidf_final = tfidf.transform(bow_final)
print(tfidf_final.shape)


# In[42]:


from sklearn.model_selection import train_test_split
x = tfidf_final
y = final_data['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[71]:


from sklearn.naive_bayes import MultinomialNB as nb


# In[72]:


fake_detect = nb().fit(x_train,y_train)


# In[73]:


fake_predict = fake_detect.predict(x_test)
print(fake_predict)


# In[74]:


from sklearn.metrics import classification_report as classy
accuracy = classy(y_test,fake_predict)
print(accuracy)



x=1
while x==1:
    x = int(input("press 1 to continue or 0 to exit :"))
    print(x)
    if x<1:
        break;
    news = input("paste news article here :")
    news = punc_remo(news)
    news = stop_remo(news)
    news = stem(news)
    bow1 = CountVectorizer().fit([news])
    bow2 = bow.transform([news])
    tfidf1 = TfidfTransformer().fit(bow2)
    tfidf2 = tfidf1.transform(bow2)
    prediction = fake_detect.predict(tfidf2)
    print("\n")
    if prediction==0:
        print("Fake news")
    
    else:
        print("Real news")
    






