import streamlit as st
import time
import pandas as pd
import altair as alt
from multipage import MultiPage
from transformers import pipeline

def app():
    st.markdown('## Mask Fill task')
    st.write('Write a sentence with a [MASK] gap to fill')
    st.markdown('## ')
    

    @st.cache(allow_output_mutation=True, suppress_st_warning =True, show_spinner=False)
    def get_model(model):
        return pipeline('fill-mask', model=model)

    def create_graph(answer):
        x_bar = [i['token_str'] for i in answer]
        y_bar = [i['score'] for i in answer]
        chart_data = pd.DataFrame(y_bar, index=x_bar)
        data = pd.melt(chart_data.reset_index(), id_vars=["index"])
        # Horizontal stacked bar chart
        chart = (
            alt.Chart(data)
            .mark_bar(color='#d7abf5')
            .encode(
                x=alt.X("index", type="nominal", title='',sort=alt.EncodingSortField(field="index", op="count", order='ascending')),
                y=alt.Y("value", type="quantitative", title="Score", sort='-x'),
            )
        )
        st.altair_chart(chart, use_container_width=True)

        
    col1, col2 = st.columns([2,1])


    with col1:
        prompt= st.text_area('Your prompt here',
            '''Who is Elon [MASK]?''') 
            
    with col2:
        select_model = st.radio(
            "Select the model to use:",
            ('Bert cased', 'Bert Un-cased'), index = 1)

        if select_model == 'Bert cased':
            model = 'bert-base-cased'
        elif select_model == 'Bert Un-cased':
            model = 'bert-base-uncased'

        with st.spinner('Loading Model... (This may take a while)'):
            unmasker = get_model(model)    
            st.success('Model loaded correctly!')
        
            gen = st.info('Generating Mask...')
            answer = unmasker(prompt)     
            gen.empty()    

    with col1:   
        create_graph(answer)       


