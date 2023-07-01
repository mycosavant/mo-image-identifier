from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential

"""
This code replaces the previous model architecture from training_v2 with MobileNetV2, a pre-trained model available in Keras.
MobileNetV2 is known for its efficiency and performance in image classification tasks.
I've made the necessary changes to incorporate MobileNetV2 into the code, including modifying the architecture and using transfer learning.

1. Imported the MobileNetV2 model from tensorflow.keras.applications.
2. Replaced the layers.Conv2D and layers.MaxPooling2D layers with the base_model obtained from MobileNetV2. This allows us to use the pre-trained MobileNetV2 model as a feature extractor.
3. Added a layers.GlobalAveragePooling2D layer after the base_model to reduce the spatial dimensions of the features.
4. Removed the previous layers.Flatten layer since it's no longer needed.
4. Changed the number of input channels from 1 to 3 in the data_augmentation block to match the RGB format of the images.
6. Disabled the training of the base_model by setting base_model.trainable = False to use the pre-trained weights.
7. Adjusted the model summary to reflect the changes in the model architecture.
These changes enable the use of the MobileNetV2 pre-trained model as a feature extractor for the image classification task.
"""



class TrainV1:
    import numpy as np
    import tensorflow as tf
    import pathlib

    dataset_url = "https://mo.columbari.us/static/images.tgz"
    data_dir = tf.keras.utils.get_file('', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    armillaria_tabescens = list(data_dir.glob('armillaria_tabescens/*'))

    batch_size = 32
    img_height = 500
    img_width = 375

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalized_ds = train_ds.map(lambda x, y: (layers.experimental.preprocessing.Rescaling(1. / 255)(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    print(np.min(first_image), np.max(first_image))

    num_classes = 36

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    """## Compile and train the model"""

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    mush1_url = "https://www.mushroomexpert.com/images/kuo6/armillaria_tabescens_06.jpg"
    mush1_path = tf.keras.utils.get_file('armillaria_tabescens_06', origin=mush1_url)

    img = keras.preprocessing.image.load_img(
        mush1_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence." .format(class_names[np.argmax(score)],
                                                                                         100 * np.max(score))
    )
