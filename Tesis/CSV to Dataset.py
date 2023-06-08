import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/Dataset.csv', header=None)

# Rename the columns
df.columns = ['aymara', 'english']

# Split the data into source and target languages
source_lang = df['aymara']
target_lang = df['english']

# Save the data into separate text files
source_lang.to_csv('source.txt', index=False, header=False)
target_lang.to_csv('target.txt', index=False, header=False)
