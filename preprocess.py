import tensorflow as tf
import os

INPUT_DIR = "images_raw"
OUTPUT_DIR = "images_augmented"
AUG_PER_IMAGE = 2

def augment_image(image_path, output_dir):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)

    img = tf.image.convert_image_dtype(img, tf.uint8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, "aug_" + os.path.basename(image_path)
    )

    tf.io.write_file(out_path, tf.image.encode_jpeg(img))


def process_class(class_name):
    in_dir = os.path.join(INPUT_DIR, class_name)
    out_dir = os.path.join(OUTPUT_DIR, class_name)

    for file in os.listdir(in_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(in_dir, file)

            for i in range(AUG_PER_IMAGE):
                augment_image(img_path, out_dir)

    print(f"✔ Done class: {class_name}")


if __name__ == "__main__":
    for cls in os.listdir(INPUT_DIR):
        if os.path.isdir(os.path.join(INPUT_DIR, cls)):
            process_class(cls)

    print("🎉 Data augmentation hoàn tất!")
