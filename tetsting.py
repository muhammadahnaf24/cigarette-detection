from ultralytics import YOLO

def main():
    model = YOLO("rokok15.pt")  # sesuaikan path model
    data_yaml = "datasetrokok15/data.yaml"             # sesuaikan path yaml
    metrics = model.val(data=data_yaml, split='test')   # atau split='test' jika ingin test set
    print(metrics)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
