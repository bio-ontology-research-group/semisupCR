import sys
#from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from biobert_embedding.embedding import BiobertEmbedding
import pandas as pd

corpus={}

infile="/ibex/scratch/kafkass/semisup_disease/sample2.txt"
#infile="/ibex/scratch/kafkass/semisup_disease/ncbi.train.lex.rep.txt"
with open(infile,'r') as f:
   for line in f:
     line = line.rstrip("\n")
     pmid,text=line.split('\t')
     corpus[pmid]=text.lower()

# Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
biobert = BiobertEmbedding()
myvectors=[]
dis_tokens=[]
neg_vectors=[]
neg_tokens=[]
pos_vectors=[]
pos_tokens=[]

for abst in corpus:
  indexlist=[]
  listt=[]
  neg_index=[]
  size=len(biobert.process_text(corpus[abst]))
  #print('sentence has {} tokens'.format(size))
  if (size<513):
     word_embeddings = biobert.word_vector(corpus[abst])
  else:
     listt = biobert.process_text(corpus[abst])[:510]
     listt.append('[SEP]')
     word_embeddings = biobert.word_vector_trun(listt)
  #word_embeddings = biobert.word_vector(corpus[abst])
  tokens=biobert.tokens
  matchers = ['mesh','doid','omim']
  for t in tokens:
   if ((matchers[0] in t) or (matchers[1] in t) or (matchers[2] in t)):
         indexlist.append(tokens.index(t))
         dis_tokens.append(t)
   else:
       neg_tokens.append(t)
       neg_index.append(tokens.index(t))

  for m in indexlist:
    #print (str(tokens[m]) + "\t" + str(word_embeddings[m].numpy()))
    myvectors.append(word_embeddings[m].numpy())
  
  for m in neg_index:
    neg_vectors.append(word_embeddings[m].numpy())

k=2

km = KMeans(n_clusters=k)
km = km.fit(myvectors)
y_kmeans=km.predict(myvectors)

u_labels = np.unique(y_kmeans)


uniq_dis={}
ratio_dic={}
df = pd.DataFrame()
freq_dic={}

df['tokens']  = dis_tokens
df['label']  = y_kmeans

#find which cluster is pos, which one is negative
for i in u_labels:
  #print ("I="+str(i))
  uniq_dis[i] = len(set(df.loc[df['label'] == i, 'tokens'].values.tolist())) #uniq tokens
if uniq_dis[0] > uniq_dis[1]:
  pos=0
  neg=1
else:
  pos=1
  neg=0
for elem in dis_tokens:
 if y_kmeans[dis_tokens.index(elem)] == pos:
   pos_vectors.append(myvectors[dis_tokens.index(elem)])
   pos_tokens.append(elem)
 else:
   neg_vectors.append(myvectors[dis_tokens.index(elem)])
   neg_tokens.append(elem)

print ("POSITIVE")
print (str(pos_tokens))
print (str(pos_vectors))

print ("NEGATIVE")
print (str(neg_tokens))
print (str(neg_vectors))

  #print ("UNIQ SET:")
  #print (set(df.loc[df['label'] == i, 'tokens'].values.tolist()))
  #print ("Uniq Frequency:"+str (uniq_dis[i]))
  #freq_dic[i] = len(df.loc[df['label'] == i].values.tolist())
  #print ("Frequency:"+str (freq_dic[i]))
  #ratio_dic[i]=float(float(uniq_dis[i])/float(freq_dic[i]))
  #print (str(i)+"\tRatio="+str(ratio_dic[i]))
##plotting the results:
#for i in u_labels:
#  plt.scatter(data_transformed[y_kmeans == i , 0] , data_transformed[y_kmeans == i , 1] , label = i)

#plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, c='yellow', label = 'Centroids',alpha=0.5)
#plt.title('Clusters of disease embeddings')
#plt.legend(loc='best')
#fname="/ibex/scratch/kafkass/semisup_disease/"+str(k)+"clusters_nomm_train.png"
#plt.savefig(fname)


