from gensim.models import KeyedVectors
from utilities import *

resource_dir = 'data/'
embeddings_dir = "embeddings/"
embedding_filename = 'word2vec_swda_300dim.txt'
embeddings_path = embeddings_dir + embedding_filename

metadata = load_data(resource_dir + "metadata.pkl")
word_to_index = metadata['word_to_index']
embedding_dimension = 300

embeddings_index = {}

# Determine if using Word2Vec, GloVe or FastText)
# Load the embeddings from file
wordvec_type = embedding_filename.split("_")[0]
if wordvec_type == 'word2vec':
    word2vec = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
else:
    file = open(embeddings_path, encoding="utf8")
    for line in file:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    file.close()

# Keep only word embeddings in the vocabulary
embedding_matrix = np.zeros((len(word_to_index), embedding_dimension))
for word, i in word_to_index.items():

    if wordvec_type == 'word2vec':
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec[word][:embedding_dimension]
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector[:embedding_dimension]

print("------------------------------------")
print("Created", embedding_dimension, "dimensional embeddings from", embeddings_path)

data = dict(embedding_matrix=embedding_matrix)

save_data(embeddings_dir + embedding_filename.split(".")[0] + ".pkl", data)
