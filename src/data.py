from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def load_metadata(metadata_path=None):
 if metadata_path is None:
  metadata_path = PROJECT_ROOT / "data" / "metadata.csv"
 metadata_path = Path(metadata_path)
 if not metadata_path.is_absolute():
  metadata_path = PROJECT_ROOT / metadata_path
 return pd.read_csv(metadata_path)
def make_dataset(frame, image_size=224, batch_size=32, shuffle=False):
 image_paths = [
 str((PROJECT_ROOT / path).resolve())
 for path in frame["image_path"]
 ]
 labels = frame["target"].astype("float32").to_numpy()
 dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
 if shuffle:
  dataset = dataset.shuffle(
 buffer_size=min(len(frame), 10000),
 seed=42,
 reshuffle_each_iteration=True,
 )
 def load_image(path, label):
  image_bytes = tf.io.read_file(path)
  image = tf.image.decode_image(
   image_bytes,
   channels=1,
   expand_animations=False,
 )
  image.set_shape([None, None, 1])
  image = tf.image.grayscale_to_rgb(image)
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, tf.float32)
  return image, label
 dataset = dataset.map(
 load_image,
 num_parallel_calls=tf.data.AUTOTUNE,
 )
 dataset = dataset.batch(batch_size)
 dataset = dataset.prefetch(tf.data.AUTOTUNE)
 return dataset
