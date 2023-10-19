import pandas as pd

df = pd.read_csv("thurs.csv")

def process_text(text):
    words = str(text).split()
    if len(words) > 30:
        # Check if page number is not equal to 1
        if df[df['Cleaned Data'] == text]['Page Number'].values[0] != 1:
            words = words[20:-10]
    else:
        return None
    return ' '.join(words)
    
# Apply the process_text function to the 'text' column
df['Cleaned Data'] = df['Cleaned Data'].apply(process_text)

df= df.dropna()

paragraphs = df['Cleaned Data']
words_to_remove = ['vol', 'ieee','eld','well','said','also','figure','images','diagram']

# Remove the specified words and last 10 words from each paragraph
cleaned_paragraphs = []

for paragraph in paragraphs:
    # Remove the specified words
    for word in words_to_remove:
        paragraph = paragraph.replace(word, '')

    # Split the cleaned paragraph into words
    words = paragraph.split()

    # Remove the last 10 words
    cleaned_paragraph = ' '.join(words[:-10])

    cleaned_paragraphs.append(cleaned_paragraph)

df['Cleaned Data'] = cleaned_paragraphs

df.to_csv('thurs_new.csv')








