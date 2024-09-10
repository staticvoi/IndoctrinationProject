from googletrans import Translator
import pandas as pd

# Load the Excel file
data = pd.read_excel('Data_german.xlsx')

# Initialize the translator
translator = Translator()

# Function to translate text from English to German
def translate_to_german(text):
    try:
        translated = translator.translate(text, src='en', dest='de')
        return translated.text
    except Exception as e:
        return str(e)

# Apply the translation function to the 'text_en' column and store the results in the 'text_de' column
data['text_de'] = data['text_en'].apply(translate_to_german)

# Save the updated dataframe with the German translations to a new Excel file
data.to_excel('Data_german.xlsx', index=False)
