from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Softmax # type: ignore

from tensorflow.keras.layers import ( # type: ignore
    Input, Dense, LSTM, Embedding, Dropout,
    Add, RepeatVector, TimeDistributed, Activation,
    Concatenate, Multiply
)

from tensorflow.keras.optimizers import Adam # type: ignore


def define_model(vocab_size, max_length):
    # IMAGE FEATURE INPUT (4096)
    image_input = Input(shape=(4096,))
    image_dense = Dense(256, activation="relu")(image_input)
    image_repeat = RepeatVector(max_length)(image_dense)

    # TEXT INPUT
    text_input = Input(shape=(max_length,))
    text_embed = Embedding(vocab_size, 256, mask_zero=True)(text_input)

    # ATTENTION
   
    # ATTENTION SECTION
    attention = Add()([image_repeat, text_embed])
    attention = Dense(256, activation="tanh")(attention)
    attention = Dense(1, activation=None)(attention)
    
    # Activation layer ki jagah Softmax layer use karo jo axis support karti hai
    attention = Softmax(axis=1)(attention) 

    attended = Multiply()([text_embed, attention])

    # LSTM DECODER
    lstm = LSTM(256)(attended)
    output = Dense(vocab_size, activation="softmax")(lstm)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005)
    )

    return model
