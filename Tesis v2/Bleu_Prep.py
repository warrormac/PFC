output_file = 'formatted_output_translations.txt'
input_file = 'output_translations.txt'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    lines = f_in.readlines()
    for line in lines:
        if line.startswith('Translation:'):
            translation = line.strip().replace('Translation: ', '')
            f_out.write(f"Output: {translation}\n")


input_file = 'output_translations.txt'
output_file = 'formatted_pre_translations.txt'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    lines = f_in.readlines()
    for line in lines:
        if line.startswith('Input:'):
            aymara_word = line.strip().replace('Input: ', '')
            f_out.write(f"Prediction: {aymara_word}\n")
