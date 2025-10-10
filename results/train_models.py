import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

df=pd.read_csv('data/features.csv')
df['tempo']=df['tempo'].astype(str).str.replace('[','').str.replace(']','').astype(float)
X=df.drop(['filename','label'],axis=1)
y=df['label']



#standardization
scaler=StandardScaler()
X_scaled=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}") 

models={
    'Decision Tree':DecisionTreeClassifier(max_depth=5,random_state=42),
    'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),
    'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7)

}


results={}
predictions={}

for name,model in models.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)
    print(f"Accuracy:{acc:.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test,pred))
    print("\nClassification Report:")
    print(classification_report(y_test,pred,target_names=['Non-transforming','transforming']))
    results[name]=acc
    predictions[name]=pred

print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
for name,acc in results.items():
    print(f"{name}:{acc:.2%}")
fig,axes=plt.subplots(1,5,figsize=(15,4))

for idx,(name,pred) in enumerate(predictions.items()):
    cm=confusion_matrix(y_test,pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=axes[idx])
    axes[idx].set_title(f"{name}\n{results[name]:.2%}")
    axes[idx].set_ylabel('True')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png',dpi=300)
print("\nSaved visualizations to results/confusion_matrices.png")
    



