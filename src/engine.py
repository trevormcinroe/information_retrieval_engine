import pickle
import tensorflow as tf
import os
import pymongo

# Disables TF's verbose logging...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Engine:
    """
    Attributes:
        Provided:
            embeddings_file (str): filepath of the embeddings matrix
            tokenizer_file (str): filepath of the tokenizer
            model_weights_file (str): filepath to the .h5 model weights
        Managed
            embeddings (array): array of embeddings matrix
            tokenizer (tf.tokenizer): tensorflow/keras tokenizer
            siamese_model (tf.keras.Model): a tensorflow/keras model

    """
    def __init__(self, embeddings_file, tokenizer_file, model_weights_file):
        self.embeddings_file = embeddings_file
        self.tokenizer_file = tokenizer_file
        self.model_weights_file = model_weights_file

        self.embeddings = None
        self.tokenizer = None
        self.siamese_model = None

    def load_embeddings(self):
        with open(self.embeddings_file, 'rb') as file:
            self.embeddings = pickle.load(file)

    def load_tokenizer(self):
        with open(self.tokenizer_file, 'rb') as file:
            self.tokenizer = pickle.load(file)

    def make_model(self):
        # The visible layer
        left_input = tf.keras.layers.Input(shape=(None,), dtype='int32')
        right_input = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embedding_layer = tf.keras.layers.Embedding(
            len(self.embeddings),
            300,
            weights=[self.embeddings],
            #input_length=max_seq_length, is this needed?
            trainable=False
        )

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_gru = tf.keras.layers.GRU(100, name='gru', recurrent_activation='sigmoid', reset_after=True,
                                         bias_initializer=tf.keras.initializers.Constant(4.5), dropout=0.0,
                                         kernel_regularizer=None, recurrent_dropout=0.0)

        left_output = shared_gru(encoded_left)
        right_output = shared_gru(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        magru_distance = tf.keras.layers.Lambda(function=lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
                                                output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        magru = tf.keras.Model([left_input, right_input], [magru_distance])
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=1, rho=0.985, clipnorm=2.5)

        magru.compile(loss='mean_squared_error', optimizer=optimizer)

        self.siamese_model = magru

    @staticmethod
    def exponent_neg_manhattan_distance(left, right):
        """Helper function for the similarity estimate of the RNNs outputs"""
        return tf.keras.backend.exp(-tf.keras.backend.sum(tf.keras.backend.abs(left - right), axis=1, keepdims=True))

    def load_model_weights(self):
        self.siamese_model.load_weights(self.model_weights_file)

    def glasses(self, query, username=<REDACTED>, password=<REDACTED>, db=<REDACTED>, collection=<REDACTED>):
        '''A helper function to write data
        Args:
            query:
            username: str
            password: str
            db: the name of the database, str
            collection: the name of the collection, str

        Returns:
            a dict
        '''
        # Connecting to the client
        client = pymongo.MongoClient('mongodb://<REDACTED>/',
                                     username=username,
                                     password=password,
                                     authSource=db,
                                     authMechanism='SCRAM-SHA-256')

        db = client[db]
        collection = db[collection]
        return collection.find(query)

    def make_query(self, keywords):
        """

        Args:
            keywords (str): str keywords

        Returns:

        """
        keywords = keywords.split(',')

        query = {'$or': [{'keywords': {'$regex': '.*' + x + '.*', '$options': 'i'}} for x in keywords]}

        return self.glasses(query=query)
