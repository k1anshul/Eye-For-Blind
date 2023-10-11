import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model
import pickle
import matplotlib.pyplot as plt
from gtts import gTTS
from IPython import display
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# Constatnts
embedding_dim = 256
units = 512
vocab_size = 5001  # top 5,000 words +1
max_length = 31
attention_features_shape = 64

# Download Model
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

new_input = image_model.input  # write code here to get the input of the image_model
hidden_layer = image_model.layers[-1].output  # write code here to get the output of the image_model

image_features_extract_model = tf.compat.v1.keras.Model(new_input, hidden_layer)  # build the final model using both input & output layer

# Define functions to load the image and generate captions


def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_jpeg(img, channels=3)  # Use 3 channels for RGB images
	img = tf.image.resize(img, (299, 299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img, image_path


def get_caption_tokenizer(caption_tokenizer_path="./model_checkpoints/tokenizer.pkl"):
	with open(caption_tokenizer_path, "rb") as f:
		return pickle.load(f)

# Define the Encoder class


class Encoder(Model):
	def __init__(self, embed_dim):
		super(Encoder, self).__init__()
		self.dense = tf.keras.layers.Dense(embed_dim)

	def call(self, features):
		features = self.dense(features)
		features = tf.keras.activations.relu(features)
		return features

# Define the Attention model class


class Attention_model(Model):
	def __init__(self, units):
		super(Attention_model, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)
		self.units = units

	def call(self, features, hidden):
		hidden_with_time_axis = hidden[:, tf.newaxis]
		score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
		attention_weights = tf.keras.activations.softmax(self.V(score), axis=1)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights

# Define the Decoder class


class Decoder(Model):
	def __init__(self, embed_dim, units, vocab_size):
		super(Decoder, self).__init__()
		self.units = units
		self.attention = Attention_model(self.units)
		self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
		self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
		self.d1 = tf.keras.layers.Dense(self.units)
		self.d2 = tf.keras.layers.Dense(vocab_size)

	def call(self, x, features, hidden):
		context_vector, attention_weights = self.attention(features, hidden)
		embed = self.embed(x)
		embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
		output, state = self.gru(embed)
		output = self.d1(output)
		output = tf.reshape(output, (-1, output.shape[2]))
		output = self.d2(output)
		return output, state, attention_weights

	def init_state(self, batch_size):
		return tf.zeros((batch_size, self.units))


# Load the captioning model and tokenizer
def restore_model(checkpoint_path="./model_checkpoints/", vocab_size=5001, embed_dim=256):
	tokenizer = get_caption_tokenizer()
	dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
	encoder = Encoder(embed_dim)
	decoder = Decoder(embed_dim, units, vocab_size)
	optimizer = tf.keras.optimizers.Adam()

	ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	return encoder, decoder, tokenizer, image_features_extract_model


encoder, decoder, tokenizer, image_features_extract_model = restore_model()


def evaluate(image_path, encoder, decoder, tokenizer, image_features_extract_model):
	attention_plot = np.zeros((max_length, attention_features_shape))
	hidden = decoder.init_state(batch_size=1)

	temp_input = tf.expand_dims(load_image(image_path)[0], 0)
	img_tensor_val = image_features_extract_model(temp_input)
	img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

	features = encoder(img_tensor_val)

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
	result = []

	for i in range(max_length):
		predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
		attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

		predicted_id = tf.argmax(predictions[0]).numpy()
		result.append(tokenizer.index_word[predicted_id])

		if tokenizer.index_word[predicted_id] == '<end>':
			return result, attention_plot

		dec_input = tf.expand_dims([predicted_id], 0)

	attention_plot = attention_plot[:len(result), :]
	return result, attention_plot


# Streamlit UI
col1, col2, col3 = st.columns([0.2,0.9,0.2])
col2.title("👀 Eye For A Blind 👀")

col1, col2, col3 = st.columns([0.05,0.9,0.05])
col2.title("Image Captioning Application")
st.write("Upload an image and get a caption!")

# My Info
# Add a title to the sidebar
st.sidebar.title("✨ About")

# Add the user's photo
col1, col2, col3 = st.sidebar.columns([0.1,0.6,0.5])
col2.image("backup/anshul_image.jpg", use_column_width=False, width=200)

# Display the designation
col1, col2, col3 = st.sidebar.columns([0.15,0.6,0.2])
col2.markdown("# Data Scientist")

# Add links to social media sites with logos
st.sidebar.markdown("### 🚀 Connect with Me:")

# # Create a horizontal layout for social media icons
col1, col2, col3, col4 = st.sidebar.columns(4)

with col1:
	st.markdown("[![LinkedIn](https://img.icons8.com/color/48/000000/linkedin-circled.png)](https://www.linkedin.com/in/k1anshul/)")
with col2:
	st.markdown("[![GitHub](https://img.icons8.com/color/48/000000/github--v1.png)](https://github.com/k1anshul)")
with col3:
	st.markdown("[![Twitter](https://img.icons8.com/color/48/000000/twitter-circled.png)](https://twitter.com/anshul_khadse)")
with col4:
	st.markdown("[![Email](https://img.icons8.com/color/48/000000/gmail.png)](anshulkhadse2011@gmail.com)")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Create two columns for the layout
col1, col2 = st.columns(2)

if uploaded_image is not None:
	# Save the uploaded image as a temporary file
	with open("backup/temp_image.jpg", "wb") as f:
		f.write(uploaded_image.read())

	# Display the uploaded image on the left half
	image = Image.open("backup/temp_image.jpg")

	col1.image(image.resize((300,300)), caption="Uploaded Image", use_column_width=True)

	if col2.button("Generate Caption"):
		# Generate caption for the uploaded image
		result, _ = evaluate("backup/temp_image.jpg", encoder, decoder, tokenizer, image_features_extract_model)
		caption = ' '.join(result).rsplit(' ', 1)[0]
		col2.write(f"✨ GENERATED CAPTION :",)
		col2.write(f"{caption.capitalize()}",)

		# Generate audio and display on the right half
		tts = gTTS(text=caption, lang="en")
		tts.save("backup/caption.mp3")
		col2.audio("backup/caption.mp3", format="audio/mp3", start_time=0)

