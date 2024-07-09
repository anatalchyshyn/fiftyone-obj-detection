# Objects detection dataset via Fiftyone

This repository provide a usefull tool to create premarked objects detection dataset of images from any videos.
Upload the videos you want to use in "video" directory

### Steps

1. Split videos into the frames.
2. Initialize the model, used for objects detection. YOLO was using in this example, as it works fast enought and return fine results, but you can use wharever pretrained model you want.
3. Send batches of images in model and get dataframe as result.
4. Postprocess it to fiftyone format.
5. Start fiftyone session to improve the quality.

### Colab

You can also use the colab notebook - https://colab.research.google.com/drive/1EkN2LvEgFC4ZUpJl0CNnUEO9wFbJLwyD?usp=sharing
