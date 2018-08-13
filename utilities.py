import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from mpl_toolkits.axes_grid1 import make_axes_locatable


def process_transcript_txt(transcript, excluded_tags=None):
    # Special characters for ignoring i.e. <laughter>
    special_chars = {'<', '>', '(', ')', '#'}

    utterances = []
    labels = []

    for utt in transcript.utterances:

        utterance = []

        for word in utt.text_words(filter_disfluency=True):

            # Remove the annotations that filter_disfluency does not (i.e. <laughter>)
            if all(char not in special_chars for char in word):
                utterance.append(word)

        # Join words for complete sentence
        utterance_sentence = " ".join(utterance)

        # Print original and processed utterances
        # print(utt.transcript_index, " ", utt.text_words(filter_disfluency=True), " ", utt.damsl_act_tag())
        # print(utt.transcript_index, " ", utterance_sentence, " ", utt.damsl_act_tag())

        # Check we are not adding an empty utterance (i.e. because it was just <laughter>)
        if len(utterance) > 0 and utt.damsl_act_tag() not in excluded_tags:
            utterances.append(utterance_sentence)
            labels.append(utt.damsl_act_tag())

    transcript_data = dict(
        utterances=utterances,
        labels=labels)

    return transcript_data


def process_batch_to_txt_file(corpus, resource_path, batch_name, excluded_tags=None):
    utterances = []
    labels = []

    batch_list = None
    if batch_name.lower() != 'all':
        # Load training or test split
        batch_list = read_file(resource_path + batch_name.lower() + "_split.txt")

    # For each transcript
    for transcript in corpus.iter_transcripts(display_progress=False):

        transcript_num = str(transcript.utterances[0].conversation_no)

        # Process if in the specified batch_name list
        if batch_list and transcript_num not in batch_list:
            continue

        transcript_data = process_transcript_txt(transcript, excluded_tags)

        # Set data values
        utterances += transcript_data['utterances']
        labels += transcript_data['labels']

        with open(resource_path + batch_name + "_text.txt", 'w+') as file:
            for i in range(len(utterances)):
                file.write(str(utterances[i]) + "|" + str(labels[i]) + "\n")


def generate_embeddings(data, metadata, verbose=False):
    word_to_index = metadata['word_to_index']
    max_utterance_len = metadata['max_utterance_len']

    label_to_index = metadata['label_to_index']
    num_labels = metadata['num_labels']

    utterances = data['utterances']
    labels = data['labels']

    tmp_utterance_embeddings = []
    tmp_label_embeddings = []

    # Convert each word and label into its numerical representation
    for i in range(len(utterances)):

        tmp_utt = []
        for word in utterances[i]:
            tmp_utt.append(word_to_index[word])

        tmp_utterance_embeddings.append(tmp_utt)
        tmp_label_embeddings.append(label_to_index[labels[i]])

    # For Keras LSTM must pad the sequences to same length and return a numpy array
    utterance_embeddings = pad_sequences(tmp_utterance_embeddings, maxlen=max_utterance_len, padding='post', value=0.0)

    # Convert labels to one hot vectors
    label_embeddings = to_categorical(np.asarray(tmp_label_embeddings), num_classes=num_labels)

    if verbose:
        print("------------------------------------")
        print("Created utterance/label embeddings, and padded utterances...")
        print("Number of utterances: ", utterance_embeddings.shape[0])

    return utterance_embeddings, label_embeddings


def generate_probabilistic_embeddings(data, frequency_data, metadata, verbose=False):
    freq_words = frequency_data['freq_words']
    probability_matrix = frequency_data['probability_matrix']

    word_to_index = metadata['word_to_index']
    max_utterance_len = metadata['max_utterance_len']

    label_to_index = metadata['label_to_index']
    num_labels = metadata['num_labels']

    utterances = data['utterances']
    labels = data['labels']

    tmp_label_embeddings = []

    # Convert each word and label into its numerical representation
    utterance_embeddings = np.zeros((len(utterances), max_utterance_len, num_labels))
    for i in range(len(utterances)):

        for j in range(len(utterances[i])):
            word = utterances[i][j]
            if word in freq_words:
                utterance_embeddings[i][j] = probability_matrix[word_to_index[word]]

        tmp_label_embeddings.append(label_to_index[labels[i]])

    # Convert labels to one hot vectors
    label_embeddings = to_categorical(np.asarray(tmp_label_embeddings), num_classes=num_labels)

    if verbose:
        print("------------------------------------")
        print("Created utterance/label embeddings, and padded utterances...")
        print("Number of utterances: ", utterance_embeddings.shape[0])

    return utterance_embeddings, label_embeddings


def batch_prediction(model, data, data_x, data_y, metadata, batch_size, verbose=False):
    # Predictions results
    correct = 0
    incorrect = 0
    correct_labels = {}
    incorrect_labels = {}
    index_to_label = metadata["index_to_label"]
    for i in range(len(index_to_label)):
        correct_labels[index_to_label[i]] = 0
        incorrect_labels[index_to_label[i]] = 0

    # Get utterance and label data
    utterances = data['utterances']
    labels = data['labels']

    # Get predictions
    predictions = model.predict(data_x, batch_size=batch_size, verbose=verbose)
    num_predictions = len(predictions)

    for i in range(num_predictions):

        # Prediction result
        prediction_result = False

        # Get prediction with highest probability
        prediction = index_to_label[np.argmax(predictions[i])]

        # Determine if correct and increase counts
        if prediction == labels[i]:
            prediction_result = True

        if prediction_result:
            correct += 1
            correct_labels[labels[i]] += 1
        else:
            incorrect += 1
            incorrect_labels[labels[i]] += 1

        if verbose:
            print("------------------------------------")
            print("Making prediction for utterance: ", utterances[i], "with label: ", labels[i])
            print("Utterance embedding: ", data_x[i])
            label_index = 0
            for j in range(len(data_y[i])):
                if data_y[i][j] > 0:
                    label_index = i
            print("Label embedding: ", label_index)
            print("Raw predictions: ", predictions)
            print("Actual label: ", labels[i])
            print("Predicted label: ", prediction)
            print("Prediction is: ", prediction_result)

            print("------------------------------------")
            print("Prediction ratios:")
            for k in range(len(index_to_label)):
                print('{:10}'.format(index_to_label[k]), " ", '{:10}'.format(correct_labels[index_to_label[k]]), " ",
                      '{:10}'.format(incorrect_labels[index_to_label[k]]))

    percent_correct = (100 / num_predictions) * correct
    percent_incorrect = (100 / num_predictions) * incorrect

    print("------------------------------------")
    print("Made ", num_predictions, " predictions")
    print("Correct: ", correct, " ", percent_correct, "%")
    print("Incorrect: ", incorrect, " ", percent_incorrect, "%")

    return predictions


def generate_confusion_matrix(data, predictions, metadata, verbose=False):
    # Get label data
    labels = data['labels']

    # Get metadata
    index_to_label = metadata['index_to_label']
    label_to_index = metadata['label_to_index']
    num_labels = metadata['num_labels']

    # Create empty confusion matrix
    confusion_matrix = np.zeros(shape=(num_labels, num_labels), dtype=int)

    # For each prediction
    for i in range(len(predictions)):
        # Get prediction with highest probability
        prediction = np.argmax(predictions[i])

        # Add to matrix
        confusion_matrix[label_to_index[labels[i]]][prediction] += 1

    if verbose:
        # Print confusion matrix
        print("------------------------------------")
        print("Confusion Matrix:")
        print('{:15}'.format(" "), end='')
        for j in range(confusion_matrix.shape[1]):
            print('{:15}'.format(index_to_label[j]), end='')
        print()
        for j in range(confusion_matrix.shape[0]):
            print('{:15}'.format(index_to_label[j]), end='')
            print('\n'.join([''.join(['{:10}'.format(item) for item in confusion_matrix[j]])]))

    return confusion_matrix


def plot_history(history, title='History'):
    # Create figure and title
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    fig.suptitle(title, fontsize=14)

    # Plot accuracy
    acc = fig.add_subplot(121)
    acc.plot(history['acc'])
    acc.plot(history['val_acc'])
    acc.set_ylabel('Accuracy')
    acc.set_xlabel('Epoch')

    # Plot loss
    loss = fig.add_subplot(122)
    loss.plot(history['loss'])
    loss.plot(history['val_loss'])
    loss.set_ylabel('Loss')
    loss.set_xlabel('Epoch')
    loss.legend(['Train', 'Test'], loc='upper right')

    # Adjust layout to fit title
    fig.tight_layout()
    fig.subplots_adjust(top=0.15)

    return fig


def plot_confusion_matrix(matrix, classes,  title='', matrix_size=10, normalize=False, color='black', cmap='viridis'):

    # Number of elements of matrix to show
    if matrix_size:
        matrix = matrix[:matrix_size, :matrix_size]
        classes = classes[:matrix_size]

    # Normalize input matrix values
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        value_format = '.2f'
    else:
        value_format = 'd'

    # Create figure with two axis and a colour bar
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

    # Generate axis and image
    ax, im = plot_matrix_axis(matrix, ax, classes, title, value_format, color=color, cmap=cmap)

    # Add colour bar
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    color_bar = fig.colorbar(im, cax=colorbar_ax)
    # Tick color
    color_bar.ax.yaxis.set_tick_params(color=color)
    # Tick labels
    plt.setp(plt.getp(color_bar.ax.axes, 'yticklabels'), color=color)
    # Edge color
    color_bar.outline.set_edgecolor(color)

    # Set layout
    fig.tight_layout()

    return fig


def plot_confusion_matrices(matrix_a, matrix_b, classes, title_a='', title_b='', matrix_size=10, normalize=False, color='black', cmap='viridis'):

    # Number of elements of matrix to show
    if matrix_size:
        matrix_a = matrix_a[:matrix_size, :matrix_size]
        matrix_b = matrix_b[:matrix_size, :matrix_size]
        classes = classes[:matrix_size]

    # Normalize input matrix values
    if normalize:
        matrix_a = matrix_a.astype('float') / matrix_a.sum(axis=1)[:, np.newaxis]
        matrix_b = matrix_b.astype('float') / matrix_b.sum(axis=1)[:, np.newaxis]
        value_format = '.2f'
    else:
        value_format = 'd'

    # Create figure with two axis and a colour bar
    fig, (ax, ax2, colorbar_ax) = plt.subplots(ncols=3, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1, 0.05]})

    # Generate axis and image
    ax, im = plot_matrix_axis(matrix_a, ax, classes, title_a, value_format, color=color, cmap=cmap)
    ax2, im2 = plot_matrix_axis(matrix_b, ax2, classes, title_b, value_format, color=color, cmap=cmap)

    # Add colour bar
    fig.colorbar(im, cax=colorbar_ax)
    color_bar = fig.colorbar(im, cax=colorbar_ax)
    # Tick color
    color_bar.ax.yaxis.set_tick_params(color=color)
    # Tick labels
    plt.setp(plt.getp(color_bar.ax.axes, 'yticklabels'), color=color)
    # Edge color
    color_bar.outline.set_edgecolor(color)

    # Set layout
    fig.tight_layout()

    return fig


def plot_matrix_axis(matrix, axis, classes, title='', value_format='d', color='black', cmap='viridis'):

    # Create axis image
    im = axis.imshow(matrix, interpolation='nearest', cmap=cmap)

    # Set title
    axis.set_title(title, color=color)

    # Create tick marks and labels
    axis.set_xticks(np.arange(len(classes)))
    axis.set_yticks(np.arange(len(classes)))
    axis.set_xticklabels(classes, color=color)
    axis.set_yticklabels(classes, color=color)
    axis.tick_params(color=color)

    # Set axis labels
    axis.set_ylabel("Actual", color=color)
    axis.set_xlabel("Predicted", color=color)

    # Rotate the tick labels and set their alignment.
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid
    for edge, spine in axis.spines.items():
        spine.set_visible(False)
    axis.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
    axis.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)
    axis.grid(which="minor", color='w', linestyle='-', linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    # Threshold determines colour of cell labels
    thresh = matrix.max() / 2.
    # Loop over data dimensions and create text annotations
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        axis.text(j, i, format(matrix[i, j], value_format),
                  ha="center", va="center",
                  color="white" if matrix[i, j] < thresh else "black")

    return axis, im


def read_file(path, verbose=True):
    with open(path, "r") as file:
        # Read a line and strip newline char
        results_list = [line.rstrip('\r\n') for line in file.readlines()]
    if verbose:
        print("Loaded data from file %s." % path)
    return results_list


def save_data(path, data, verbose=True):
    file = open(path, "wb")
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    if verbose:
        print("Saved data to file %s." % path)


def load_data(path, verbose=True):
    with open(path, 'rb') as file:
        saved_data = pickle.load(file)
        file.close()
    if verbose:
        print("Loaded data from file %s." % path)
    return saved_data
