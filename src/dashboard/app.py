import streamlit as st
import requests, os

API = os.getenv('FASTAPI_URL', 'http://host.docker.internal:8000')

st.title('AI Testing Agent - Dashboard (Starter)')

st.markdown('### Recent Prediction Example')
file = st.text_input('File path', 'src/app/login.py')
if st.button('Get prediction'):
    resp = requests.post(f'{API}/predict', json={'repo':'demo/repo','file':file,'features':{}})
    st.json(resp.json())

st.markdown('### Generate Test')
if st.button('Generate sample test'):
    resp = requests.post(f'{API}/generate_test', json={'source':{'type':'demo'}})
    data = resp.json()
    st.code(data.get('code',''), language='typescript')
