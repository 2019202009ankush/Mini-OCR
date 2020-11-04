
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
class KNNClassifier:
	def __init__(self):
		self.best_k=1
		self.data_set=None
		self.dic11={}
		self.dic11['values_of_k']=[]
		self.dic11['accuracy']=[]
	def euq(self,x, y):
		if(x.size==y.size):
			x=x[1:]
			y=y[1:]
		elif (x.size<y.size):
			y=y[1:]
		dist = np.linalg.norm(x-y)
		return dist


	def make_euq_vec(self,val_arr,train_arr):
		final_list=[]
		count=0
		for i in range (0,len(val_arr)):
			lis=[]
			for j in range (0,len(train_arr)):
				lis1=[]
				lis1.append(self.euq(val_arr[i],train_arr[j]))
				lis1.append(train_arr[j][0])
				lis.append(lis1)
			lis.sort()
			# print("lis=",lis)
			# print(count,end='',flush=True)
			count+=1
			final_list.append(lis)
		# print('final_list=',final_list)
		return final_list

	def knn_match(self,equli_vec,k):
		lis=[]
		for i in range(0,len(equli_vec)):
			dic={}
			# print(equli_vec[i][0],equli_vec[i][1],equli_vec[i][2],equli_vec[i][3],equli_vec[i][4],equli_vec[i][5])
			for j in range(0,k):
				# print('j=',j)
				if equli_vec[i][j][1] in dic.keys():
					dic[equli_vec[i][j][1]]+=1
				else:
					dic[equli_vec[i][j][1]]=1
			liss1=list(dic.values())
			liss2=list(dic.keys())
			# print(dic)
			lis.append(liss2[liss1.index(max(liss1))])
		return lis

	def make_confusion_matrix(self,lis1,lis):#only to visualization
		# print('lis=',lis,'lis1=',lis1)
		list1=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
		for f, b in zip(lis1, lis):
			f=f-97
			b=b-97
			# print(f, b)
			if(f==b):
				# print('f and b equals')
				# print(list1[f][b])
				list1[f][b]=list1[f][b]+1
				# print(list1[f][b])
			else:
				list1[f][b]=list1[f][b]+1
# 		print(len(list1),len(list1[0]))
		# for i in list1:
		# 	for j in i:
		# 		# print(j,end='',flush=True)
		# 	# print('')
				
	def print_f1_score(self,lis1,lis):#only for visualization
		return f1_score(lis,lis1,average='micro')
	
	

	def accuracy(self,lis1,lis):
		count=0
		for i in range(0,len(lis1)):
			if lis[i] == lis1[i]:
				count=count+1
		# print('count=',count)
		return count/len(lis1)

	def actual_l(self,val_arr):
		lis=[]
		for i in range (0,len(val_arr)):
			lis.append(val_arr[i][0])
		return lis

	def data_part(self,data, percent):
		idx = data.index.tolist()
		r,_=data.shape
		size=int(r*percent)
		rand_idx = random.sample(population=idx, k=size)
		val_data = data.loc[rand_idx]
		train_data = data.drop(rand_idx)
		return val_data, train_data
	def convert_dataframe_to_ascii(self,data):
		for column in data:
			data[column]=data[column].apply(ord)
		return data

	def train(self,a):
# 		print('wait it can take upto 10 min to give the output')
		data=pd.read_csv(a)
		data.drop(['?'],axis=1)
		data=self.convert_dataframe_to_ascii(data)
		val_data,train_data=self.data_part(data,0.1)
# 		print("partition size is",data.shape,val_data.shape,train_data.shape)
		train_arr=train_data.to_numpy()
		val_arr=val_data.to_numpy()
		self.data_set=data.to_numpy()
		vec=self.make_euq_vec(val_arr,train_arr)
		lis1=self.actual_l(val_arr)
		dic={1:0.0,2:0.0,3:0.0,4:0.0}
		for k in range (1,9):
			# print('-----------------------------------------------------------------------------------------------------')
			lis=self.knn_match(vec,k)
			# print("myKnnPredict",lis,"original list=",lis1)
			effi=self.accuracy(lis1,lis)
			# print("accuracy vs k is accuracy= ",effi,'k=',k)
			dic[k]=effi
			# print("confusion_matrix for k=",k)
			self.make_confusion_matrix(lis1,lis)
			f1=self.print_f1_score(lis1,lis)
			# print("f1_score for k=",k,' is ',f1)
			
			# print('accuracy_score for k=',k,' is ',effi)
			self.dic11['values_of_k'].append(k)
			self.dic11['accuracy'].append(effi)
			# print('------------------------------------------------------------------------------------------------------')
# 		print(dic)
		liss1=list(dic.values())
		liss2=list(dic.keys())
		self.best_k=liss2[liss1.index(max(liss1))]
		# print('best k is' , self.best_k)
		p=pd.DataFrame.from_dict(self.dic11)
		p.plot(x='values_of_k',y='accuracy',color='red')
		plt.show()

	def predict(self,a):
		test_data=pd.read_csv(a,header=None)
		# test_data=test_data.head(100)
		# test_data.drop(['?'],axis=1)
		for column in test_data:
			test_data[column]=test_data[column].apply(ord)
		test_arr=test_data.to_numpy()
# 		print(test_data.shape)
		vec=self.make_euq_vec(test_arr,self.data_set)
		lis=self.knn_match(vec,self.best_k)
		# print('predicted lis=',lis)
		lis1=[chr(i) for i in lis]
		return lis1