import streamlit as st
# Custom imports 
from multipage import MultiPage
from pages import text_gen, translation

# Create an instance of the app 
app = MultiPage()

# Add all your applications (pages) here
#app.add_page("Home Page", home_page.app)
#app.add_page("Chat Bot", chat.app)
#app.add_page("Mask Fill", fill_mask.app)
app.add_page("Translation Pulido", text_gen.app)
app.add_page("Translation Huggin", translation.app)

# The main app
app.run()
