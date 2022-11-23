import streamlit as st
import time
from multipage import MultiPage
from transformers import pipeline
import torch


def app():
    st.markdown('## Translation task')
    st.write('Write something in English and AI will translate')
    st.markdown('## ')

    @st.cache(allow_output_mutation=True, suppress_st_warning =True, show_spinner=False)
    def get_model(model):
        return pipeline(task = 'translation', model = model)
        
    col1, col2 = st.columns([2,1])

        
    with col1:
        prompt= st.text_area('Your prompt here',
            '''What is the translation of this sentence?''') 
            
    with col2:
        model = 'Helsinki-NLP/opus-mt-en-es' 

        with st.spinner('Loading Model... (This may take a while)'):
            generator = get_model(model)    
            st.success('Model loaded correctly!')

    with col1:    
        gen = st.info('Generating text...')
        answer = generator(prompt)                 
        gen.empty()                      
                        
    lst = answer[0]['translation_text']
    
    t = st.empty()
    for i in range(len(lst)):
        t.markdown("#### %s..." % lst[0:i+1])
        time.sleep(0.04)



