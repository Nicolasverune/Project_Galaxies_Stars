import warnings
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn 
import time

#sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier

#add complexity
from sklearn.preprocessing import PolynomialFeatures

#CV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.metrics import recall_score

#Learning Curve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

#random forest
from sklearn.ensemble import RandomForestClassifier


def build_tuned_param(kernel_l,c_l, gamma_l ):
    tuned_param = []
    for kernel in kernel_l : 
        if kernel == 'linear' : 
            tuned_param.append({'kernel': [kernel],'C': c_l,'verbose':[0],'max_iter':[-1] })
        else : 
            tuned_param.append({'kernel': [kernel], 'gamma': gamma_l,'C': c_l,'verbose':[0],'max_iter':[-1]})
    return tuned_param

@ignore_warnings(category=ConvergenceWarning)
def test_svm(kernel_l,c_l, gamma_l , galaxies, stars,percentage_l,stand=False,norma= False,details=False) : 
        
    
    time_before = time.time() 
    
    result_dict = {}

    for percentage in percentage_l : 
        print('#############')  
        print('THE PERCENTAGE IS',percentage)
        print('#############') 
        gal_train, gal_test = train_test_split(galaxies, test_size = percentage)
        star_train, star_test = train_test_split(stars, test_size = percentage)

        gs_train = gal_train.append(star_train, ignore_index=True)
        gs_test = gal_test.append(star_test, ignore_index=True)
        
        gs_train_x = gs_train.drop(['type'],axis=1)
        gs_train_y = gs_train['type']    
        gs_test_x = gs_test.drop(['type'],axis=1)
        gs_test_y = gs_test['type']
        gs_test_y = gs_test_y.astype('int')
        gs_train_y = gs_train_y.astype('int')        
        
        if stand :
            ###  Standardization, or mean removal and variance scaling ###
            #scaler = preprocessing.MinMaxScaler()
            #X_test_minmax = min_max_scaler.transform(X_test)
            #
            #X_train_maxabs = max_abs_scaler.fit_transform(X_train)
            scaler = preprocessing.StandardScaler(with_mean=True , with_std= True).fit(gs_train_x)
            scaler.transform(gs_train_x)
            scaler.transform(gs_test_x)
        
        if norma : 
            ###  Normalization  ###
            gs_train_x = preprocessing.normalize(gs_train_x, norm='l2') # norm = 'l1','l2','max'
            gs_test_x = preprocessing.normalize(gs_test_x, norm='l2')
        
        #gamma only for ‘rbf’, ‘poly’ and ‘sigmoid’
        #clf = svm.SVC(kernel = type_kernel,, C = 1.0, gamma = ’scale’)
        #clf.fit(gs_train_x, gs_train_y)
        #time_after = time.time()
        
        tuned_parameters = build_tuned_param(kernel_l,c_l,gamma_l)
        scores = ['accuracy']#,'recall']

        for score in scores:
            print('-----------------------------')
            print("# Tuning hyper-parameters for %s \n" % score)

            clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring='%s' % score, n_jobs = -1)
            clf.fit(gs_train_x, gs_train_y)

            print("Best parameters set found on development set:\n")
            tempo = clf.best_params_         
            print(tempo,"\n")
            tempotuple = (tempo.get('C'),tempo.get('gamma'),tempo.get('kernel'))
            if tempotuple in result_dict.keys() : 
                result_dict[tempotuple] = result_dict.get(tempotuple) + 1 
            else : 
                result_dict[tempotuple] = 1 

            #for more details            
            if details : 
                print("Grid scores on development set:\n")
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r " % (mean, std * 2, params))
            
            ### Cross Validation ###
            #scoring = ['precision_macro', 'recall_macro']
            #clf = svm.SVC(kernel='linear', C=1, random_state=0)
            #return train score to know the score on the training set 
            scoring_train = cross_validate(clf, gs_train_x, gs_train_y, scoring=scores, return_train_score = False)
            scoring_test =  cross_validate(clf, gs_test_x, gs_test_y, scoring=scores, return_train_score = False)
            print("scoring train",scoring_train['test_'+score].mean(),"\n")
            print("scoring test",scoring_test['test_'+score].mean(),"\n")
            
            #easy cross validation 
            """
            loo = LeaveOneOut()
            lpo = LeavePOut(p=2)
                for train, test in loo.split(X):
            """

            y_pred = clf.predict(gs_test_x)
            print("confusion matrix\n",confusion_matrix(gs_test_y,y_pred,labels=np.unique(y_pred)))
            print("classification report\n",classification_report(gs_test_y,y_pred))
            
            """
            print("Detailed classification report:\n")
            print("The model is trained on the full development set.\n")
            print("The scores are computed on the full evaluation set.\n")
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            """
            
            print('-----------------------------')
        
        #tot_time = time_after - time_before
    
    return result_dict
        
            
#def n_fold_cross_validation(clf,data,nf,rn) : 


def test_learning_curve(galaxies,stars,per,random):
    
    plt.figure(figsize = (16,5))    

    gs = galaxies.append(stars, ignore_index=True)     

    gs_x = gs.drop(['type'],axis=1)
    gs_y = gs['type']    
    gs_y = gs_y.astype('int')
       
    train_sizes = np.trunc(len(gs)*np.array(per))
    
    if random : 
        estimators = [SGDClassifier(class_weight='balanced'),RandomForestClassifier(class_weight='balanced')]
    else : 
        estimators = [LinearRegression(),RandomForestRegressor()]
    
    for i in range(len(estimators)) : 
	#use scoring 'neg_mean_squared_error' for regression 
        train_sizes, train_scores, validation_scores = learning_curve(estimator = estimators[i],X = gs_x,y = gs_y, train_sizes = per, cv = 5,scoring = matthews_corrcoef(), shuffle = True)

        train_scores_mean = -train_scores.mean(axis = 1)
        validation_scores_mean = -validation_scores.mean(axis = 1)
        plt.style.use('seaborn')    
        plt.subplot(1,2,i+1)

        
        plt.plot(train_sizes, train_scores_mean, label = 'Training error')
        plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
        plt.ylabel('MSE', fontsize = 14)
        plt.xlabel('Training set size', fontsize = 14)
        plt.title('Learning curves LR model', fontsize = 18, y = 1.03)
        plt.legend()
        plt.ylim(0,3)
    
    plt.show()
    
    


def get_proper_items( file_l, n,top_random = "random", respect_percentage = True) : 
    
     #flux
    flux_l = ['type','u','g','r','i','z']
    big_file = pd.DataFrame(columns = flux_l)
    for smallfile in file_l : 
        big_file = big_file.append(pd.read_csv(smallfile).filter(['type','u','g','r','i','z'],axis=1),ignore_index=True)
    
    tot = len(big_file.index)    
    gal_n = len(big_file[big_file['type']==3].index)
    gal_p = gal_n/tot    

    if top_random == 'random':     
        #get n rows    
        if respect_percentage : 
            g = big_file[big_file['type']==3].sample(n=int(n*gal_p))
            s = big_file[big_file['type']==6].sample(n=int(n*(1-gal_p)))
            rd_gs = g.append(s, ignore_index=True)  
        else : 
            rd_gs = big_file.sample(n=n)

        rd_gs = add_complexity(rd_gs)
        return rd_gs

    elif top_random == 'top' : 
        #get n or n-1 rows
        g = big_file[big_file['type']==3].head(int(n/2))
        s = big_file[big_file['type']==6].head(int(n/2))
       
        n_gs = g.append(s, ignore_index=True)   
        n_gs = n_gs.sample(frac=1).reset_index(drop=True)  
        
        n_gs = add_complexity(n_gs)
        return n_gs

def add_complexity(gs):

    flux_l = ['type','u','g','r','i','z']
    #add colors
    for i in range(1,len(flux_l)-1) : 
        gs[flux_l[i]+"-"+flux_l[i+1]] = gs[flux_l[i]] - gs[flux_l[i+1]]

    ### Polynomial features - adding complexity ###
    poly = PolynomialFeatures(degree = 3, interaction_only=True) # interation only to not produce terms like x^2, but only x1*x2 etc..
    poly.fit_transform(gs)  
    
    return gs


def controler_test_svm(kernel_l,file_l,n,random,percentage_l,number ) : 
    
    gs = get_proper_items(file_l,n)
    
    #select gal/stars
    galaxies = gs[gs['type'] == 3]
    stars = gs[gs['type'] == 6]
    
    #usd for c and gamma
    param_l = [0.1, 1., 10.]
    
    if number ==1 :
        result = test_svm(kernel_l,param_l,param_l,galaxies,stars,percentage_l,True,True,False)
        return result
    elif number == 2 : 
        test_learning_curve(galaxies,stars,percentage_l,True) 
    elif number == 3 :
        result = test_svm(kernel_l,param_l,param_l,galaxies,stars,percentage_l,True,True,False)
        test_learning_curve(galaxies,stars,percentage_l,True) 
        return result
    

svm_type_kernel = ["linear", "poly", "rbf", "sigmoid"]
percentage_l = list(np.arange(1,10,2)/10)


result = controler_test_svm(svm_type_kernel,["ugriz_gal_star.csv"],3000,True,percentage_l,2)
print('@@')
print(result)









 

