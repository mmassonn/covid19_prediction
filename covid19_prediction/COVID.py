#Covid
#1.Definir un objectif mesurable :
    #Objectif : Predire si une personne est infectee en fonction des donnees cliniques disponibles
    #Metrique : F1 -> 50%  Score F1 -> 70%
    
#Import Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
#2.EDA(Exploraty Data Analysis)

#load data
df = pd.read_excel('dataset.xlsx')

#Shape analysis

    #All row and columns showed
    pd.set_option('display.max_row', 111) 
    pd.set_option('display.max_column', 111)
    
    #Explore first fives rows ==> df values
    df.head()
    
    #target identification : SARS-Cov-2 exam result

    #shape : 5644,111
    df.shape      
    
    #variable types
    df.dtypes.value_counts()
    
    #Nan values beaucoup de Nan (moitié) des variables >90% de NaN) et 2 groupes de données 76%-->test viral, 89%-->taux sanguins   
    sns.heatmap(df.isna(), cbar=False)
    
    (df.isna().sum()/df.shape[0]).sort_values(ascending = True)
    
    
#Data analysis
    
    #drop columns unusable
    df = df[df.columns[df.isna().sum()/df.shape[0] <0.9]] 
    df.head
    df = df.drop('Patient ID', axis=1)
    df.head

    #target vizualisation_ 10% of positive values and 90% of negative values
    df['SARS-Cov-2 exam result'].value_counts(normalize = True)
    
    #Signification des variables
        #continue variables histograms : standardisées, skewed (asymétrique), test sanguin
        for col in df.select_dtypes('float'):
            plt.figure()
            sns.distplot(df[col].dropna())
        
        #age quantile : difficile d'interpréter ce graph, clairement ces données ont été traitées, on pourrait penser 0-5 mais cela pourrait aussi être une transformation math. 
        sns.distplot(df['Patient age quantile'])
        
        #qualitative variables : binaire (0,1 et rhinovirus un peu plus élevé en détecté)
        for col in df.select_dtypes('object'):
            print(f'{col :-<50} {df[col].unique()}')
        for col in df.select_dtypes('object'):
            plt.figure()
            df[col].value_counts().plot.pie()
        
    #Target/variable ralations
    
        #Created positive and negative under set
        positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']   
        negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']       
            
        #Created blood and viral set
        missing_rate = df.isna().sum()/df.shape[0]
        blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)]
        viral_columns = df.columns[(missing_rate <0.80) & (missing_rate > 0.75)]
        
        #Taget/blood : les taux de Monocytes, Platelets, leukocytes semblent liés au covid 19 : hypothèse à tester
        for col in blood_columns:
            plt.figure()
            sns.distplot(positive_df[col].dropna(), label='positive')
            sns.distplot(negative_df[col].dropna(), label = 'negative')
            plt.legend()
        
        #Target/age : les individus de faible age sont tres peu contaminés? attention on ne connait pas l'age, et on ne sait pas de quand date le dataset (s'il s'agit des enfants on sait que les enfants sont touchés autant que les adultes) En revanche cette variable poura être intéresante pour la comparer avec les résultats de test sanguin
        sns.countplot(x='Patient age quantile',hue='SARS-Cov-2 exam result', data=df)
        
        #Target/viral : les doubles cas sont trés rares. Rhino/Enter positif -covid - 19 négatif? hypothèse à tester ? mais il est possible que la région est subit une épidémie de ce virus.
        for col in viral_columns:
            plt.figure()
            sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'],df[col]), annot=True,fmt='d')
            
            
    #Variable/variable relations
    
        #blood/blood : certaines variables sont trés corrélées : +0.9 (a surveiller plus tard)
        sns.clustermap(df[blood_columns].corr())
        
        #blood/age : trés faible corrélation
        for col in blood_columns:
            plt.figure()
            sns.lmplot(x='Patient age quantile', y=col ,hue='SARS-Cov-2 exam result', data=df)
        
        df.corr()['Patient age quantile'].sort_values()
        
        #viral/viral : influenza rapid test donne de mauvais résultats, il faudra peut être le laisser tomber
        pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])
        pd.crosstab(df['Influenza B'], df['Influenza B, rapid test']) 
        
        #viral/blood : les taux sanguin entre malade et covid 19  sont différents: hypothèse à tester
        df['est malade'] = np.sum(df[viral_columns[:-2]]=='detected', axis=1)>=1
        df.head()
        
        malade_df = df[df['est malade']== True]
        non_malade_df = df[df['est malade']== False]
        
        for col in blood_columns:
            plt.figure()
            sns.distplot(malade_df[col].dropna(), label='malade')
            sns.distplot(non_malade_df[col].dropna(), label = 'non malade')
            plt.legend()
        
        #hospitalisation/blood : intéressant dans le cas ou on devrai prédire dans quelle service un patient devrait aller
        
        def hospitalisation (df):
            if df['Patient addmited to regular ward (1=yes, 0=no)']==1:
                return 'surveillance'
            if df['Patient addmited to semi-intensive unit (1=yes, 0=no)']==1:
                return 'soins semi-intensives'
            if df['Patient addmited to intensive care unit (1=yes, 0=no)']==1:
                return 'soins intensifs'
            else:
                return 'inconnu'
            
            
        df['statut'] = df.apply(hospitalisation, axis=1)
        df.head()
        
        for col in blood_columns:
            plt.figure()
            for cat in df['statut'].unique():
                sns.distplot(df[df['statut']==cat][col].dropna(), label=cat)
            plt.legend()
            
        
        
    #Nan analyse viral: 1350(92/8), blood: 600(87/13), both : 90
    df.dropna().count()
    df[blood_columns].count()
    df[viral_columns].count()
    
    df1 = df[viral_columns[:-2]]
    df1['covid'] = df['SARS-Cov-2 exam result']
    df1.dropna()['covid'].value_counts(normalize=True)
    
    df2 = df[blood_columns[:-2]]
    df2['covid'] = df['SARS-Cov-2 exam result']
    df2.dropna()['covid'].value_counts(normalize=True)
    
    #null hypothèsis (H0):
    
    #Les individus atteinds du covid-19 ont des taux de leucocytes, monocytes, platelets significativement différents
        
        #H0 = les taux moyens sont égaux chez les individus positifs et négatifs.
        #T-test
        from scipy.stats import ttest_ind
        #normalement autant de positif que de négatif mais ce n'est pas le cas
        positive_df.shape
        negative_df.shape
        
        balanced_neg = negative_df.sample(positive_df.shape[0])
        
        #t-test
        def t_test(col):
            alpha =0.02
            stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
            if p< alpha:
                return 'HO rejetée'
            else: 
                return 0
        
        for col in blood_columns :
            print (f'{col :-<50} {t_test(col)}')
        

#3.Pre-processing
            
    
    #Variable selection
    missing_rate = df.isna().sum()/df.shape[0]
    
    blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)])
    viral_columns = list(df.columns[(missing_rate <0.80) & (missing_rate > 0.75)])
    
    Key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']
    
    df = df[ Key_columns + blood_columns + viral_columns]
    df.shape
    df.head()
    
    
    #Split Train and Test set       
    from sklearn.model_selection import train_test_split 
    trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
   
    trainset['SARS-Cov-2 exam result'].value_counts()
    testset['SARS-Cov-2 exam result'].value_counts()
    
    
    #Encodage
    def encodage(df):
        
        code = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
        
        for col in df.select_dtypes('object') :
            df[col] = df[col].map(code)
        
        return df
        
    df.dtypes.value_counts()
    
    def feature_engineering (df):
        df['est malade'] = df[viral_columns].sum(axis=1)>=1
        df = df.drop(viral_columns, axis = 1)
        
        return df
    
    def imputation(df):
        
#        df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leucocytes'].isna())
#        df.fillna(-999)
        df = df.dropna(axis=0)
        
        return df
    
    def preprocessing(df):
        
        df = encodage(df)
        df = feature_engineering (df)
        df = imputation(df)
        
        X = df.drop('SARS-Cov-2 exam result', axis =1)
        y = df['SARS-Cov-2 exam result']
        
        print(y.value_counts())
        
        return X,y
    
    X_train, y_train = preprocessing(trainset)
    X_test, y_test = preprocessing(testset)
        
    
#4.Modelling  
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif#Annova
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

preprocessor= make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))#nombre de variable à modifier

RandomForest = make_pipeline(preprocessor, RandomForestClassifier (random_state = 0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier (random_state = 0))
SVM = make_pipeline(preprocessor, StandardScaler(),SVC (random_state = 0))
KNN = make_pipeline(preprocessor, StandardScaler(),KNeighborsClassifier ())

dict_of_models = {'RandomForest' : RandomForest,'AdaBoost' : AdaBoost,'SVM': SVM,'KNN': KNN}

for name, model in dict_of_models.items():
    print(name)
    evaluation(model)

#5.Procédure d'évaluation
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

def evaluation(model):
    
    model.fit(X_train, y_train)
    
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N,train_score,val_score = learning_curve(model, X_train, y_train, cv=4, scoring ='f1', train_sizes= np.linspace(0.1,1,10))
    
    plt.figure(figsize=(12,8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='val score')
    plt.legend()
    
    
evaluation(model)

pd.DataFrame(model.feature_importances_, index=X_train.columns).plot.bar(figsize = (12,8))

#Optimisation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
SVM

hyper_params = {'svc__gamma' : [1e-3, 1e-4], 'svc__C' : [1, 10, 100, 1000],'pipeline__polynomialfeatures__degree' :[2,3,4], 'pipeline__selectkbest__k' : range(40,60)}

grid = RandomizedSearchCV(SVM, hyper_params, scoring ='recall', cv=4, n_iter=40)

grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))

evaluation(grid.best_estimator_)

#Prediction recall  curve
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()

def model_final(model, X, threshold = 0):
    return model.decision_function(X) > threshold

y_pred : model_final(grid.best_estimator_, X_test, threshold=-1)

f1_score(y_test, y_pred)

recall_score(y_test, y_pred)