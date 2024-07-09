from detector import *
from utils import *

# init variables
video_folder = '/content/video/*'
output_folder = 'output_frames'
batch_size = 64
val_size = 0.2
model_type = 'ultralytics/yolov5'
model_name = 'yolov5m'
model_classes = [0, 2, 5, 7, 14, 15, 16, 17, 18, 19]

# get frames and init the model
od = ObjectDetection(video_folder, output_folder)
od.video_processing()
model = od.load_model(model_type, model_name, model_classes)

# procced frames throw the model and get dataframe with detected objects
batch_list = od.batch_list(glob.glob(f"/content/{output_folder}/*"), batch_size)
batch_df = od.model_step(model, batch_list)
combined_df = pd.concat(batch_df, ignore_index=True)

# split on train and validation and print objects distribution
train_set, val_set = split_to_train_and_val(combined_df, val_size)
show_class_distribution(train_set, val_set)

# upload in fiftyone
train_samples = to_fiftyone(train_set, 'train')
val_samples = to_fiftyone(val_set, 'validation')
dataset = fo.Dataset("classes-detection")
dataset.add_samples(train_samples)
dataset.add_samples(val_samples)

# start its session
session = fo.launch_app(dataset)