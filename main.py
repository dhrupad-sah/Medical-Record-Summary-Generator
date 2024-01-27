from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

def remove_duplicate_sentences_in_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = infile.read()

    paragraphs = data.split("----------------------------------------------------------------------------------------------------")

    unique_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            sentences = paragraph.split('. ')
            unique_sentences = set()
            output_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    if sentence not in unique_sentences:
                        unique_sentences.add(sentence)
                        output_sentences.append(sentence)

            unique_paragraph = '. '.join(output_sentences)
            unique_paragraphs.append(unique_paragraph)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for unique_paragraph in unique_paragraphs:
            outfile.write(unique_paragraph + '\n' + '-'*100 + '\n')

def filter_entities(output, desired_entity_groups, excluded_symptoms):
    return [
        entity for entity in output
        if entity['entity_group'] in desired_entity_groups and entity['word'].lower() not in excluded_symptoms
    ]

def convert_to_text(entities):
    text_output = ""
    for entity in entities:
        text_output += f"{entity['entity_group']}: {entity['word']}\n"
    return text_output

remove_duplicate_sentences_in_file('Medical_Records.txt', 'Pre-processed.txt')

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

with open('Pre-processed.txt', 'r') as file:
    data_examples = file.read().split('----------------------------------------------------------------------------------------------------')

desired_entity_groups = ['Age', 'Clinical_event', 'Date', 'Sign_symptom', 'Medication']
excluded_symptoms = ['di', '##zziness', 'symptoms']

with open('Summarized_Records.txt', 'a') as combined_file:
    for example in data_examples:
        output = pipe(example)

        filtered_output = filter_entities(output, desired_entity_groups, excluded_symptoms)

        text_output = convert_to_text(filtered_output)

        combined_file.write(f"Summary: {example}\n")
        combined_file.write(text_output + '\n' + '-'*100 + '\n')

if os.path.exists('Pre-processed.txt'):
    os.remove('Pre-processed.txt')
    print(f"The file {'Pre-processed.txt'} has been deleted.")
else:
    print(f"The file {'Pre-processed.txt'} does not exist.")
