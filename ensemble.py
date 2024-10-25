#                                                               ENSEMBLE LEARNİNG

# Ensemble learning, makine öğrenmesinde birden fazla modelin bir araya getirilerek kullanıldığı bir tekniktir. 

# VotingClassifier    AdaBoost   Random Forest ensemble learning algoritması
#%% kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.datasets import make_moons,make_circles,make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier

import warnings
warnings.filterwarnings("ignore")

#%%
n_samples = 2000
n_features = 10
n_classes = 2
random_state = 42

noise_class = 0.3
noise_moon = 0.3
noise_circle = 0.3

# multi classification için kullanıcam
X,y = make_classification(n_samples=n_samples,                # 100 farklı veri var
                    n_features=n_features,                    # kaç değişkene sahip kaç özelliği olsun
                    n_classes = n_classes,                    # veri setinde kaç sınıf olsun
                    n_repeated=0,                             # aynı özellikte veriler olmasın
                    n_redundant=0,                            # anlamsız sample sayısı
                    n_informative= n_features-1,              # Bu, bilgilendirici özelliklerin sayısını belirler. Yani, toplam özellik sayısından bir eksiğini alır
                    random_state=random_state,
                    n_clusters_per_class=1,
                    flip_y = noise_class)                     # 0.2 verdiğimi için verilerin %20sini rastgele  karıştırır amaç daha karmaşık ve gerçekçi bir yapıda olmasına yardımcı olur ve modelin dayanıklılığını test etmek için

# kafamıza göre veriseti yaratamıyoruz buradaki default değerlerini iyi ayarlamamız gerekiyor bu yüzden bu hata çıkıyor
# ValueError: n_classes(2) * n_clusters_per_class(2) must be smaller or equal 2**n_informative(1)=2
# hatanın çözümü için dediği gibi 2ye eşit olmalı veya 2 den küçük olmalı
# nedeni default hali per_class 2 alıyor  2üssü 2 =4 ama 1 verirsek 2 üssü 1 = 2 .
# n_clusters_per_class=1 yaparak sorunu çözebiliriz

# kodumuz çalıştı X benim featurlarım yani benim algoritmaları train edeceğim X trainim    y ise labellar


data=pd.DataFrame(X)
data["target"] = y
plt.figure()
sns.scatterplot(x=data.iloc[:,0],y=data.iloc[:,1],hue="target",data=data)

# mesela n_class sayısını arttırdım 3 yaptım  ve yine hata n_classes(3) * n_clusters_per_class(1) must be smaller or equal 2**n_informative(1)=2
# bunu çözmemiz için n_features sayısını arttırmamız gerekiyor

data_classification = (X,y)

moon = make_moons(n_samples=n_samples,                  # binary classification için kullanıcam
                  noise=noise_moon,                     #  Veri noktalarının yerlerini rastgele değiştirir 
                                                        # amaç daha karmaşık ve gerçekçi bir yapıda olmasına yardımcı olur ve modelin dayanıklılığını test etmek için
                  random_state=random_state) 

datamoon = pd.DataFrame(moon[0])
datamoon["target"] = moon[1]
plt.figure()
sns.scatterplot(x=datamoon.iloc[:,0],y=datamoon.iloc[:,1],hue="target",data=datamoon)


# binary classification için kullanıcam
circle = make_circles(n_samples=n_samples,  
                      factor= 0.1,                      # içteki dairenin çapının dıştaki daireye göre boyut oranını belirler
                      noise=noise_circle,
                      random_state=random_state)


circledata = pd.DataFrame(circle[0])
circledata["target"] = circle[1]
plt.figure()
sns.scatterplot(x=circledata[0],y=circledata[1],hue="target",data=circledata)

datasets = [moon,circle]
#%%
n_estimators = 10

scv = SVC()
knn = KNeighborsClassifier(n_neighbors=15)
dt = DecisionTreeClassifier(random_state=random_state,max_depth=2)
rf=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state,max_depth=2)
ada = AdaBoostClassifier(base_estimator=dt,n_estimators=n_estimators,random_state=random_state)
vt = VotingClassifier(estimators=[("SVC",scv),("KNN",knn),("DT",dt),("RF",rf),("ADA",ada)])   # default modu hard

names = ["SVC","KNN","Decision Tree","Random Forest","Ada Boost","Voting Classifier"]

classifier = [scv,knn,dt,rf,ada,vt]
h = 0.2
i = 1
figure = plt.figure(figsize=(24,12))


for ds_cnt,ds in enumerate(datasets):
    X,y = ds
    X=RobustScaler().fit_transform(X)
    # Robustscaler veri ölçekleyicidir ve verileri ölçeklendirmek için kullanılır. Ancak diğer ölçekleme yöntemlerinden
    # farklı olarak, aşırı uç (outlier) değerlerden daha az etkilenir.
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=random_state)
    
    x_min,x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min,y_max = X[:,1].min() - .5, X[:,1].max() + .5
    
    xx,yy= np.meshgrid(np.arange(x_min,x_max,h),
                       np.arange(y_min,y_max,h))   # anlamı şöyle gibi  min ve max değerleri al verileri bunların içine yerleştir

   
    cm = plt.cm.RdBu # anlamı renk kırmızıdan maviye gidiyor
    cm_bright = ListedColormap(["#FF0000","#0000FF"])
    
    ax = plt.subplot(len(datasets),len(classifier) +1,i)
    if ds_cnt == 0:
        ax.set_title("İnput data")
    ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,edgecolors="k")
    ax.scatter(X_test[:,0],X_test[:,1],c=y_test,alpha=0.6,cmap=cm_bright,edgecolors="k")
    i+=1
    print("Dataset : ",ds_cnt)
    
    
    for name,clf in zip(names,classifier):
        ax = plt.subplot(len(datasets),len(classifier) +1,i)
        
        clf.fit(X_train,y_train)
        
        score = clf.score(X_test, y_test)
        print(f"{name}: Test Score : {score}")
        
        score_train = clf.score(X_train, y_train)
        print(f"{name}: Train Score : {score_train}")
        print()
        
        if hasattr(clf,"decision_function"):
            Z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()])   # xx.ravel(),yy.ravel() çok boyutlu bir diziyi tek boyutlu bir diziye dönüştürür
# decision_function, her (x, y) koordinatı için karar sınırına olan uzaklığı hesaplar.Bu sayede karar sınırlarının nerede olduğunu anlayabilir ve görselleştirebiliriz.
        else:
            Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])           # xx.ravel(),yy.ravel() çok boyutlu bir diziyi tek boyutlu bir diziye dönüştürür
            # np.c_[xx.ravel(), yy.ravel()] ifadesi, ızgara üzerindeki her (x, y) koordinat çiftini birleştirir ve bunları decision_function fonksiyonuna sokar.
        Z = Z.reshape(xx.shape)
        ax.contourf(xx,yy,Z,cmap=cm,alpha=0.8)  # veri setinizde sınıflar arasında karar sınırlarını belirgin hale getirir
        
        ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,edgecolors="k")
        ax.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=cm_bright,edgecolors="k",alpha=0.6,marker="+")
        
        ax.set_xticks(())
        ax.set_yticks(())
        # Y ekseninde gösterilecek işaretleyicileri ayarlar. Yine, parantez içinde boş bir tuple kullanarak, Y eksenindeki işaretleyicilerin gizlenmesini sağlar.
        
        if ds_cnt ==0:
            ax.set_title(name)
        score = score*100
        ax.text(xx.max() - .3 ,yy.min() + .3 ,("%.1f"%score),
                size=15,horizontalalignment="right")
        i+=1
        print("---------------------------------------------------------------------")

plt.tight_layout()
plt.show()


def make_classify(dc,clf,name):
    x,y=dc
    x = RobustScaler().fit_transform(x)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.4,random_state=random_state)
    
    for name,clf in zip(names,classifier):
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        print(f"{name}: Test Score : {score}")
        score_train = clf.score(X_train, y_train)
        print(f"{name}: Train Score : {score_train}")
        print()
print("  DATASET  #   2 ")
make_classify(data_classification,classifier,names)

#%%


#               Adaptive Boosting (AdaBoost), makine öğreniminde kullanılan bir topluluk öğrenme (ensemble learning) tekniğidir

# w: Her bir örneğin (gözlemin) başlangıçta aldığı ağırlığı temsil eder.
# n: Eğitim setindeki toplam örnek sayısını temsil eder.

# w =  1 / n
#            yani bizim mesela 100 verimiz var   w = 1/100


# 1. kuralım bu .  weightleri initialize(ağırlıkların başlangıç değerlerini belirlemek) ettikten sonra yapmamız gereken
# 2.  error weight diye parametre hesaplayabilmek için
# yanlış olan wrongweightleri toplamamız gerekiyor   formül # W = Σ wi    i aslında wnun sağ altında


#3. adım en küçük olan error rate seçicem , seçtikten sonra votingpower diye parametrem var onu hesaplıycam

# 4. adım
# Alpha (α) Formülü # Formül  :    α = (1/2) * log((1 - E) / E )

#  α: Zayıf sınıflandırıcının ağırlığıdır. Bu, sınıflandırıcının genel model üzerindeki katkısını belirler.
#  E: Zayıf sınıflandırıcının hata oranıdır. Yani, sınıflandırıcının yanlış tahmin ettiği örneklerin.        hesaplıyoruz

# hesapladıktan sonra weightleri güncellemem gerekiyor ve modeli oluşturmam gerekiyor
# weightleri güncellerken eski veya doğru olduğuna bakmaksızın güncellemem gerekiyor
# W(yeni) = W(eski) / 2(1-E)  = correct ise 
# W(eski) / 2E = Wrong ise

# 5. adım
# modeli oluştur
# # h(x) = (1/n) * Σ (from i=1'den n kadar ) f(x_i)

# 5. adımı güncelledikten sonra tekrar 4. adımdan  2. adıma dönüyoruz

# stump dediğimiz şey aslında basit bir decision tree
# stumpları kullanarak hx modelini elde etmeye çalışıcaz











#%% voting classifier

# mesela bir resmimiz var kedi ise 1 köpek ise 0 diyecek yapay zekalar
#               SVM                 KNN             ADABOOST        DT      RF
#RESMİ           1                   1                  1            0       0
#YOLLADIK

# 3 TANE 1       2 TANE 0  VAR   3 > 2 'den bu bir kedi diyoruz buna votingclassifier'da hard parametresini kullanarak yapabiliriz
#                      soft votingclassifier'da bana sınıf bilgisini net dönmüyor ama köpek olma olasılığı %70 kedi olma %30 diyor




















