import nltk
from utilities import *

resource_dir = 'data/'
embeddings_dir = "embeddings/"

# Threshold for minimum number of words to include in the matrix
freq_thresh = 2

# Split into labels and sentences
text_data = read_file(resource_dir + "all_text.txt")
sentences = []
da_tags = []
for line in text_data:
    sentences.append(line.split("|")[0])
    da_tags.append(line.split("|")[1])

# Load metadata
metadata = load_data(resource_dir + "metadata.pkl")
word_to_index = metadata['word_to_index']
index_to_word = metadata['index_to_word']
label_to_index = metadata['label_to_index']
index_to_label = metadata['index_to_label']

vocabulary = metadata['vocabulary']
print("Vocabulary Size: ", len(vocabulary))

labels = metadata['labels']
print("Number of  labels: ", len(labels))

# Get words >= threshold
freq_words = []
for word in vocabulary:
    if word[1] >= freq_thresh:
        freq_words.append(word[0])

print("Number of words over frequency threshold: ", len(freq_words))

# Generate word count matrix
word_count_matrix = np.zeros((len(freq_words), len(labels)), dtype=int)
for i in range(len(sentences)):

    sentence_tokens = nltk.word_tokenize(sentences[i])

    for word in sentence_tokens:
        if word in freq_words:
            word_count_matrix[word_to_index[word]][label_to_index[da_tags[i]]] += 1

# Print word count matrix
# print('{:20}'.format("words"), end='')
# for i in range(freq_matrix.shape[1]):
#     print('{:10}'.format(labels[i][0]), end='')
# print()
# for i in range(freq_matrix.shape[0]):
#     print('{:15}'.format(freq_words[i]), end='')
#     print('\n'.join([''.join(['{:10}'.format(item) for item in freq_matrix[i]])]))

# Calculate probability matrix
probability_matrix = np.zeros((len(freq_words), len(labels)))
for i in range(probability_matrix.shape[0]):
    word_count = vocabulary[i][1]

    for j in range(probability_matrix.shape[1]):
        probability_matrix[i][j] = (100 / word_count) * word_count_matrix[i][j]

# Print probability matrix
# print('{:20}'.format("words"), end='')
# for i in range(probability_matrix.shape[1]):
#     print('{:10}'.format(labels[i][0]), end='')
# print()
# for i in range(probability_matrix.shape[0]):
#     print('{:15}'.format(freq_words[i]), end='')
#     print('\n'.join([''.join(['{:10.2f}'.format(item) for item in probability_matrix[i]])]))

# Save data to file
data = dict(
    freq_words=freq_words,
    probability_matrix=probability_matrix)

save_data(embeddings_dir + "probabilistic_freq_" + str(freq_thresh) + ".pkl", data)