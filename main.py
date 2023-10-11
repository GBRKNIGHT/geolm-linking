import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd

# Set the option to display all columns
pd.options.display.max_columns = 5

# Model name from Hugging Face model hub
model_name = "zekun-li/geolm-base-toponym-recognition"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Example input sentence
input_sentence = "Minneapolis, officially the City of Minneapolis, is a city in " \
                 "the state of Minnesota and the county seat of Hennepin County."
input_2 = "Los Angeles, often referred to by its initials L.A., is the most populous " \
          "city in California, the most populous U.S. state. It is the commercial, " \
          "financial,  and cultural center of Southern California. Los Angeles is the " \
          "second-most populous city in the United States after New York City, with a population of " \
          "roughly 3.9  million residents within the city limits as of 2020."

# input_sentence = input_2

# Tokenize input sentence
tokens = tokenizer.encode(input_sentence, return_tensors="pt")
original_words = tokenizer.convert_ids_to_tokens(tokens[0])
# tokenizer.to
# Pass tokens through the model
outputs = model(tokens)

# Retrieve predicted labels for each token
predicted_labels = torch.argmax(outputs.logits, dim=2)

predicted_labels = predicted_labels.detach().cpu().numpy()
# Decode predicted labels
predicted_labels = [model.config.id2label[label] for label in predicted_labels[0]]

# Print predicted labels
print(predicted_labels)
# ['O', 'B-Topo', 'O', 'O', 'O', 'O', 'O', 'B-Topo', 'O', 'O', 'O', 'O', 'O', 'O',
# 'O', 'O', 'B-Topo', 'O', 'O', 'O', 'O', 'O', 'B-Topo', 'I-Topo', 'I-Topo', 'O', 'O', 'O']

name_list = []  # store the place where B-topo emerged. \
place = 0
# this for loop find place where B-Topo emerged,
for i in predicted_labels:
    if i == "B-Topo":
        name_list.append(place)
    place = place + 1

# this for loop finds if I-topo emerged after the B-topo emerged.
name_length_list = []
j = 1
for i in name_list:
    while predicted_labels[i + j]:
        if predicted_labels[i + j] == "I-Topo":
            j = j + 1
        else:
            name_length_list.append(j)
            j = 1
            break

# find the word according to name_list and name_length_list
print(original_words)
print(name_list)
print(name_length_list)

# this part merge I-topo to B-topo.
which_word = 0
for length in name_length_list:
    if length == 1:
        which_word += 1
        continue
    else:
        start_topo = original_words[name_list[which_word]]
        i = 1
        while i < length:
            start_topo = start_topo + original_words[name_list[which_word] + i]
            i += 1
        original_words[name_list[which_word]] = start_topo
        which_word += 1
print(original_words)

# This part find all words and delete '#'
all_words = []
i = 0
while i < len(name_list):
    word = original_words[name_list[i]]
    word = word.replace("#", "")
    # this loop add a space before a uppercase letter
    word_length = len(word)
    j = 1
    while j < word_length:
        if word[j].isupper() & (word[j - 1].isalpha()):
            word = word[:j] + ' ' + word[j:]
        j += 1
    all_words.append(word)
    i += 1
print(all_words)

# ['O', 'B-Topo', 'O', 'O', 'O', 'O', 'O', 'B-Topo', 'O', 'O', 'O', 'O', 'O', 'O',
# 'O', 'O', 'B-Topo', 'O', 'O', 'O', 'O', 'O', 'B-Topo', 'I-Topo', 'I-Topo', 'O', 'O', 'O']
# what happend to other 0s #

dtypes_dict = {
    0: int,  # geonameid
    1: str,  # name
    2: str,  # asciiname
    3: str,  # alternatenames
    4: float,  # latitude
    5: float,  # longitude
    6: str,  # feature class
    7: str,  # feature code
    8: str,  # country code
    9: str,  # cc2
    10: str,  # admin1 code
    11: str,  # admin2 code
    12: str,  # admin3 code
    13: str,  # admin4 code
    14: int,  # population
    15: int,  # elevation
    16: int,  # dem (digital elevation model)
    17: str,  # timezone
    18: str  # modification date yyyy-MM-dd
}

# Load the Geonames dataset into a Pandas DataFrame
geonames_df = pd.read_csv('cities5000.txt', sep='\t', header=None,
                          names=['geonameid', 'name', 'asciiname', 'alternatenames',
                                 'latitude', 'longitude', 'feature class', 'feature code',
                                 'country code', 'cc2', 'admin1 code', 'admin2 code',
                                 'admin3 code', 'admin4 code', 'population', 'elevation',
                                 'dem', 'timezone', 'modification date'])

# print(geonames_df)

# create 2-d matrix to store lines.
total_words = len(geonames_df)
print(total_words)

# String array to compare
string_array_to_compare = all_words

# Create a filter using isin() to check if 'name', 'asciiname', or 'alternatenames' match any string in the array
filter_condition = (geonames_df['name'].isin(string_array_to_compare) |
                    geonames_df['asciiname'].isin(string_array_to_compare) |
                    geonames_df['alternatenames'].apply(lambda x: any(substring in x for substring in string_array_to_compare) if isinstance(x, str) else False))


# Apply the filter to the DataFrame
filtered_df = geonames_df[filter_condition]

# Print the filtered DataFrame
print(filtered_df)

print(filtered_df['alternatenames'].to_csv(index=False))
