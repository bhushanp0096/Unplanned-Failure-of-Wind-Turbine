# Import libraries
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import pickle, joblib
import numpy as np

# Load the saved model
imputer = joblib.load(r"Imputation")
winsor = joblib.load(r"Winsorization")
scalar = joblib.load(r"Scaling")
MNB_model = pickle.load(open(r"MultinomialNB_GS.pkl", 'rb'))

def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    duplicates = data.drop_duplicates(subset=data.columns.difference(["date"]), keep=False)
    data1 = duplicates.drop(["date"], axis=1)
    
    imputation = pd.DataFrame(imputer.transform(data1), columns=data1.columns)
    outlier_treatment = pd.DataFrame(winsor.transform(imputation), columns=imputation.columns)
    scaling = pd.DataFrame(scalar.transform(outlier_treatment), columns=outlier_treatment.columns)
    MNB_prediction = pd.DataFrame(MNB_model.predict(scaling), columns=['Failure_Status_Predicted'])
    final = pd.concat([MNB_prediction , data1], axis=1)
    final['Failure_Status_Predicted'] = np.where(final['Failure_Status_Predicted'] == 1, 'Failure', final['Failure_Status_Predicted'])
    final['Failure_Status_Predicted'] = np.where(final['Failure_Status_Predicted'] == '0', 'No_Failure', final['Failure_Status_Predicted'])

    
    final.to_sql('wind_turbine_failure_prediction', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final


def main():  

    st.title("Wind Turbine")
    st.sidebar.title("Dataset & Database Credentials")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Wind Turbine Gearbox Failure Prediction</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Enter DataBase Credientials Here:</p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here" ,type="password")
    pw = st.sidebar.text_input("password", "Type Here" ,type="password")
    db = st.sidebar.text_input("database", "Type Here" )
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
                                   
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))
                           
if __name__=='__main__':
    main()


