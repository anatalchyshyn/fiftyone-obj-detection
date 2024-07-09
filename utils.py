def show_class_distribution(train_set, val_set):
    
    # print distribution of classes and sets
    
    print("Training Set")
    print("Images: ", train_set.shape[0])
    print(train_set['name'].value_counts())
    print("------------------")
    print("\nValidation Set")
    print("Images: ", val_set.shape[0])
    print(val_set['name'].value_counts())

def split_to_train_and_val(df, val_size):
    
    # split dataset to trained and validation with balanced class values
    
    grouped = df.groupby('name')
    train_data = []
    val_data = []

    for _, group in grouped:

        train_group, val_group = train_test_split(group, test_size=val_size, random_state=123)
        train_data.append(train_group)
        val_data.append(val_group)

    train_set = pd.concat(train_data)
    val_set = pd.concat(val_data)
    return train_set, val_set


def to_fiftyone(combined_df, tag):

    # preprocess to fiftyone format
    
    samples = []
    for path in combined_df['filepath'].unique():
        sample = fo.Sample(filepath=path, tags=[tag])

        detections = []
        sliced = combined_df[combined_df['filepath']==path]
        for _, row in sliced.iterrows():
            label = row['name']
            bounding_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            detections.append(fo.Detection(label=label, bounding_box=bounding_box))

        sample["ground_truth"] = fo.Detections(detections=detections)

        samples.append(sample)

    return samples