import nltk
from operator import itemgetter
from swda import CorpusReader
from gensim.models import Word2Vec
from utilities import *

batch_name = 'all'
resource_dir = 'data/'
embeddings_dir = "embeddings/"
text_file_path = resource_dir + "all_text.txt"
corpus = CorpusReader('switchboard_data/')

# Excluded dialogue act tags
excluded_tags = ['x', '+']

# Dimension for switchboard embeddings
embedding_dimension = 300

# Process switchboard csv's
process_batch_to_txt_file(corpus, resource_dir, batch_name, excluded_tags=excluded_tags)

print("Processing file: ", text_file_path)
text_data = read_file(text_file_path)

# Split into labels and sentences
sentences = []
da_tags = []
for line in text_data:
    sentences.append(line.split("|")[0])
    da_tags.append(line.split("|")[1])

# Generate tokenised utterances
max_utterance_len = 0
utterances = []

for sentence in sentences:

    current_utterance = nltk.word_tokenize(sentence)

    # Determine maximum utterance length
    if len(current_utterance) > max_utterance_len:
        max_utterance_len = len(current_utterance)

    utterances.append(current_utterance)

# Count total number of utterances
num_utterances = len(utterances)

# Count the words and frequencies
word_frequency = nltk.FreqDist(itertools.chain(*utterances))

# Generate vocabulary
vocabulary = word_frequency.most_common()
vocabulary_size = len(vocabulary)

# Create index-to-word and word-to-index
index_to_word = dict()
word_to_index = dict()
for i, word in enumerate(vocabulary):
    index_to_word[i] = word[0]
    word_to_index[word[0]] = i

# Write frequencies and enumerations to file
with open(resource_dir + 'words.txt', 'w+') as file:
    for j in range(vocabulary_size):
        file.write(str(word_to_index[vocabulary[j][0]]) + " " + str(index_to_word[j]) + " " + str(vocabulary[j][1]) + "\n")

# Generate Word2Vec embeddings for switchboard
word2vec = Word2Vec(utterances, size=embedding_dimension, window=5, min_count=1, workers=2)
word2vec.wv.save_word2vec_format(embeddings_dir + "word2vec_swda" + str(embedding_dimension) + "_dim.txt", binary=False)

# Count the labels and frequencies
labels = nltk.defaultdict(int)
for tag in da_tags:
    labels[tag] += 1

labels = sorted(labels.items(), key=itemgetter(1), reverse=True)

# Count number of labels
num_labels = len(labels)

# Create index-to-label and label-to-index
index_to_label = dict()
label_to_index = dict()
for i, label in enumerate(labels):
    index_to_label[i] = label[0]
    label_to_index[label[0]] = i

# Write label frequencies and enumerations to file
with open(resource_dir + 'labels.txt', 'w+') as file:
    for k in range(num_labels):
        file.write(str(label_to_index[labels[k][0]]) + " " + str(index_to_label[k]) + " " + str(labels[k][1]) + "\n")

print("------------------------------------")
print("Created vocabulary and word/label indexes for switchboard data...")
print("Max utterance length: ", max_utterance_len)
print("Vocabulary size: ", vocabulary_size)

# Save data to file
data = dict(
    num_utterances=num_utterances,
    max_utterance_len=max_utterance_len,
    vocabulary=vocabulary,
    vocabulary_size=vocabulary_size,
    index_to_word=index_to_word,
    word_to_index=word_to_index,
    labels=labels,
    num_labels=num_labels,
    label_to_index=label_to_index,
    index_to_label=index_to_label)

save_data(resource_dir + "metadata.pkl", data)
