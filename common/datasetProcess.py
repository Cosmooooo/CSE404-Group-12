import os, csv, json

def data_process(root, csv_file):
    data = []
    labels = dict()
    files = dict()
    for file_name in os.listdir(root):
        file = file_name.split(".")[0]
        _, video_id, clip_id, first_name, last_name = file.split("_")
        file = "_".join([first_name, last_name, video_id, clip_id])
        data.append(file)
        files[file] = os.path.join(root, file_name)

    with open (csv_file) as f:
        reader = csv.reader(f)
        row = next(reader)
        for row in reader:
            name, label = row[0].split(".")[0], row[1:]
            _, first_name, last_name, video_id, clip_id = name.split("_")
            labels["_".join([first_name, last_name, video_id, clip_id])] = [float(l) for l in label[:4]]
    return data, labels, files


if __name__ == "__main__":
    root = "/media/cosmo/Dataset/YTCelebrity/ytcelebrity/"
    csv_file = "../celebrity.csv"
    data, labels, files = data_process(root, csv_file)
    
    path_label = dict()

    for d in data:
        path_label[files[d]] = labels[d]

    with open("../data.json", "w") as f:
        json.dump(path_label, f)
