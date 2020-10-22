import streamlit as st
import pickle

lin_model=pickle.load(open('learner_KNN.pkl','rb'))
model_linear=pickle.load(open('model_linear.pkl','rb'))
model_logistic=pickle.load(open('model_logistic.pkl','rb'))
model_SVM=pickle.load(open('model_SVM.pkl','rb'))
model_tree=pickle.load(open('model_tree.pkl','rb'))


def classify(num):
    if num == 1:
        return 'Heart Risk!'
    elif num == 0:
        return 'No Heart risk!'
    else:
        return 'Exception'

def main():
    st.title("Heart Disease Classification")
    html_temp = """
    <div style="background-color:teal ; padding:10px">
    <h2 style="color:white; text-align:center;">Heart Disease Prediction </h2>
    </div>
    """
    
    
    
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['KNN','SVM','Random Forest']
    option=st.sidebar.selectbox('Select the model',activities)
    st.subheader(option)
    st.spinner("Hello")
    
    age=st.text_input('Age')
    sex=st.text_input('Select sex (1-Male, 0-Female)')
    cp=st.slider('Select cp',0.0,10.0)
    trestbps=st.slider('Select trestbps',0.0,200.0)
    chol=st.slider('Select chol',0.0,400.0)
    fbs=st.text_input('Select fbs (0-Normal, 1-Diabetes)')
    restecg=st.text_input('Select restecg (0-Normal, 1-having ST-T, 2-hypertrophy')
    thalach=st.slider('Select thalach',0.0,300.0)
    exang=st.text_input('Select exang')
    oldpeak=st.slider('Select oldpeak',0.0,5.0)
    slope=st.slider('Select Sepal slope',0.0,2.0)
    ca=st.slider('Select ca',0.0,2.0)
    thal=st.slider('Select thal',1,4)
    
    inputs=[[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    
    if st.button('Classify'):#button name is Classify
        if option == 'KNN':
            st.success(classify(lin_model.predict(inputs)))
        elif option == 'SVM':
            st.success(classify(model_SVM.predict(inputs)))
        else:
            st.success(classify(model_tree.predict(inputs)))
            
                
            

if __name__=='__main__':
    main()
        
    
            
    
    