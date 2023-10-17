def setup_dataset(args):
    # setup dataset as pandas data frame
    dataset = getattr(__import__("datasets.%s" % args.dataset), args.dataset)
    dataset_root = "./data/%s" % args.dataset
    if args.dataset_root is not None:
        dataset_root = args.dataset_root
    df_dict = dataset.setup_df(dataset_root)

    dataset_dict = {}
    # key is train/val/test and the value is corresponding pytorch dataset
    for split, df in df_dict.items():
        target_transform = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
        # target_transform is mapping from category name to category idx start from 0
        if split == "train":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset_dict[split] = ImagePandasDataset(df=df,
                                                 img_key="path",
                                                 label_key="label",
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 )
    return dataset_dict


def setup_dataloader(args, dataset_dic):
    dataloader_dict = {}
    episodes_dict = {"train": args.episodes_train, "val": args.episodes_val, "test": args.episodes_test}
    for split, dataset in dataset_dic.items():
        episodes = episodes_dict[split]

        if split == "train":
            nway = args.nway
        else:
            nway = args.nway_eval
        dataloader_dict[split] = DataLoader(
            dataset,
            batch_sampler=NShotTaskSampler(
                dataset,
                episodes,
                args.nshot,
                nway,
                args.nquery,
            ),
            num_workers=args.workers,
        )
    return dataloader_dict