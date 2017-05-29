import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.datasets import fetch_20newsgroups
py.sign_in('omkar.9194', '5m1uZZg49qiyDSGe4oSq')

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
 
# Loading the data of all caegories
twenty_train = fetch_20newsgroups(subset='all',categories=categories)

# Getting the counts of each category
counter=0
counts_categories_arr = [] 
while counter < len(categories) :
    twenty_train = fetch_20newsgroups(subset='all',categories=categories[counter].split())
    count, = twenty_train.target.shape
    counts_categories_arr.append(count)
    counter+=1

    

#Plotting the histogram
histogram = [
    go.Bar(
        x=categories,
        y=counts_categories_arr
    )
]
plot_url = py.plot(histogram, filename='categories_histogram')

#Calculating the count of 'Computer Technology'
comp_cat_count = 0
for i in range(1,6):
    comp_cat_count = comp_cat_count + counts_categories_arr[i]
print ("Count of Computer Technology documents :- ",comp_cat_count)

#Calculating the count of 'Recreational Activity'
rec_cat_count = 0
for i in range(7,11):
    rec_cat_count = rec_cat_count + counts_categories_arr[i]
print ('Count of Recrreational Activity documents :- ', rec_cat_count)