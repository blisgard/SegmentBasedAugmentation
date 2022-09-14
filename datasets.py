# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

#LIBRARIES #
import warnings
import numpy as np
import pandas as pd
import copy
import torch
import random
import cv2
from torch.utils.data import Dataset
from PIL import Image
import transforms
warnings.filterwarnings("ignore")

"""============================================================================"""
# FUNCTION TO RETURN ALL DATALOADERS NECESSARY #
import time

def to_seconds(date):
    return time.mktime(date.timetuple())

def give_dataloaders(opt):
    """
    Args:
        dataset: string, name of dataset for which the dataloaders should be returned.
        opt:     argparse.Namespace, contains all training-specific parameters.
    Returns:
        dataloaders: dict of dataloaders for training, testing and evaluation on training.
    """
    datasets = give_datasets(opt)

    # Move datasets to dataloaders.
    dataloaders = {}
    for key, dataset in datasets.items():
        if isinstance(dataset, customDataset) and key == 'training':
            # important: use a SequentialSampler
            # see reasoning in class definition of SuperLabelTrainDataset
            dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
                                                           num_workers=opt.kernels,
                                                           sampler=torch.utils.data.SequentialSampler(dataset),
                                                           pin_memory=True, drop_last=False)
        else:
            is_val = dataset.is_validation
            dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
                                                           num_workers=opt.kernels, shuffle=not is_val, pin_memory=True,
                                                           drop_last=not is_val)

    return dataloaders


def give_datasets(opt):
    """
    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath = opt.source_path
    image_sourcepath_test = opt.source_path + "/images"
    # Load text-files containing classes and imagepaths
    training_files = pd.read_table(opt.source_path + '/Info_Files/metu_trademark_train.txt', header=0,
                                   delimiter='\t')
    test_files = pd.read_table(opt.source_path + '/Info_Files/logo_test.txt', header=0, delimiter='\t')
    query_files = pd.read_table(opt.source_path + '/Info_Files/logo_query.txt', header=0, delimiter='\t')
    query_test_files = pd.read_table(opt.source_path + '/Info_Files/test_query_imagepaths.txt', header=0, delimiter='\t')
    # Generate Conversion dict.
    conversion = {}
    for class_id, path in zip(training_files['class_id'], training_files['path']):
        conversion[class_id] = path.split("/")[0]
    for class_id, path in zip(test_files['class_id'], test_files['path']):
        conversion[class_id] = path.split("/")[0]
    for class_id, path in zip(query_files['class_id'], query_files['path']):
        conversion[class_id] = path.split("/")[0]
    for class_id, path in zip(query_test_files['class_id'], query_test_files['path']):
        conversion[class_id] = path.split("/")[0]
    # Generate image_dicts of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict, val_image_dict = {}, {}
    for key, img_path in zip(training_files['class_id'], training_files['path']):
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath + '/' + img_path)
    for key, img_path in zip(test_files['class_id'], test_files['path']):
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath + '/' + img_path)
    query_image_dict = {}
    for key, img_path in zip(query_files['class_id'], query_files['path']):
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(image_sourcepath_test + '/' + img_path)
    quert_test_image_dict = {}
    for key, img_path in zip(query_test_files['class_id'], query_test_files['path']):
        if not key in quert_test_image_dict.keys():
            quert_test_image_dict[key] = []
        quert_test_image_dict[key].append(image_sourcepath_test + '/' + img_path)

    train_dataset = customDataset(train_image_dict, query_image_dict, opt)
    query_dataset = BaseTripletDataset(query_image_dict, opt, is_validation=True)
    query_test_dataset = BaseTripletDataset(quert_test_image_dict, opt, is_validation=True)
    val_dataset = BaseTripletDataset(val_image_dict, opt, is_validation=True)
    eval_dataset = BaseTripletDataset(train_image_dict, opt, is_validation=True)

    train_dataset.conversion = conversion
    val_dataset.conversion = conversion
    eval_dataset.conversion = conversion
    query_dataset.conversion = conversion
    query_test_dataset.conversion = conversion
    return {'training': train_dataset, 'testing_gallery': val_dataset, 'evaluation': eval_dataset,
            'query': query_test_dataset}


class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """

    def __init__(self, image_dict, opt, samples_per_class=4, is_validation=False):
        """
        Dataset Init-Function.
        Args:
            image_dict:         dict, Dictionary of shape {class_idx:[list of paths to images belong to this class] ...} providing all the training paths and classes.
            opt:                argparse.Namespace, contains all training-specific parameters.
            samples_per_class:  Number of samples to draw from one class before moving to the next when filling the batch.
            is_validation:      If is true, dataset properties for validation/testing are used instead of ones for training.
        Returns:
            Nothing!
        """
        # Define length of dataset
        self.n_files = np.sum([len(image_dict[key]) for key in image_dict.keys()])
        self.is_validation = is_validation
        self.pars = opt
        self.image_dict = image_dict
        self.avail_classes = sorted(list(self.image_dict.keys()))
        # Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict = {key: self.image_dict[key] for i, key in enumerate(self.avail_classes)}

        self.avail_classes = sorted(list(self.image_dict.keys()))
        # Init. properties that are used when filling up batches.
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            # Select current class to sample images from up to <samples_per_class>
            self.current_class = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        # Data augmentation/processing methods.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            if opt.resize256:
                transf_list.extend([transforms.Resize(256)])
            transf_list.extend([transforms.RandomResizedCrop(size=224),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize((224,224))])

        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        # Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        self.image_list = [[(x, key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        # Flag that denotes if dataset is called for the first time.
        self.is_init = True

    def ensure_3dim(self, img):
        """
        Function that ensures that the input img is three-dimensional.
        Args:
            img: PIL.Image, image which is to be checked for three-dimensionality (i.e. if some images are black-and-white in an otherwise coloured dataset).
        Returns:
            Checked PIL.Image img.
        """
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (sample_class, torch.Tensor() of input image)
        """
        if self.is_init:
            self.current_class = self.avail_classes[idx % len(self.avail_classes)]
            self.is_init = False
        if not self.is_validation:

            if self.samples_per_class == 1:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

            if (self.samples_per_class == 0 and self.n_samples_drawn == len(self.image_dict[self.current_class])
                    or self.n_samples_drawn == self.samples_per_class):
                # Once enough samples per class have been drawn, we choose another class to draw samples from.
                # Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                # previously or one before that.
                # NOTE: if self.samples_per_class is 0, then use all the images from current_class
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter:
                        counter.remove(prev_class)

                self.current_class = counter[idx % len(counter)]
                self.classes_visited = self.classes_visited[1:] + [self.current_class]
                self.n_samples_drawn = 0

            class_sample_idx = idx % len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1

            out_img = self.transform(
                self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
            return self.current_class, out_img
        else:
            return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files



flatten = lambda l: [item for sublist in l for item in sublist]

class customDataset(Dataset):
    def __init__(self, image_dict, query_image_dict, opt):
        self.image_dict = image_dict
        self.query_image_dict = query_image_dict
        self.dataset_name = opt.dataset
        self.batch_size = opt.bs
        self.samples_per_class = opt.samples_per_class
        self.new_image_dict = {}
        self.a = []
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((instance, sub))
            self.new_image_dict[sub] = newsub
        for sub in self.query_image_dict:
            newsub = []
            for instance in self.query_image_dict[sub]:
                newsub.append((instance, sub))
            self.new_image_dict[sub] = newsub

        # checks
        # provide avail_classes
        self.avail_classes = [*self.new_image_dict]
        # Data augmentation/processing methods.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transf_list = []

        transf_list.extend([transforms.SegmentColor(), transforms.Resize((224,224))])
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform_query = transforms.Compose(transf_list)
        transf_list = []

        transf_list.extend([transforms.Resize(256),
            transforms.RandomResizedCrop((224, 224)), transforms.RandomHorizontalFlip(0.7),
            transforms.RandomVerticalFlip(0.75)],
        )
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        self.image_list = [[(x, key) for x in self.new_image_dict[key]] for key in self.new_image_dict.keys()]
        for i in self.image_list:
            for j in i:
                self.a.append(j[0])
        self.image_list = self.a
        self.reshuffle()
        self.makeBatchesBinaryLabelled()

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def reshuffle(self):

        new_image_dict = copy.deepcopy(self.new_image_dict)
        for sub in new_image_dict:
            random.shuffle(new_image_dict[sub])

        count = 0
        classes = [*new_image_dict]

        total_batches = []
        batch = []
        finished = 0
        temp = []
        tempi=[]
        while finished == 0:
            random_choice = random.choice(classes[1:])
            classes_temp = [random_choice, 0]
            len_of_batch = 0
            for sub_class in classes_temp:
                if sub_class != 0:
                    length = len(new_image_dict[sub_class])
                    if length >= 4:
                        x = random.sample(new_image_dict[sub_class], 4)
                        batch.append(x)
                        temp.append(x)
                        random.shuffle(new_image_dict[sub_class])
                    else:
                        if length >= 1:
                            y = random.sample(new_image_dict[sub_class], length)
                            batch.append(y)
                            temp.append(y)
                    len_of_batch += 4
                else:
                    if len_of_batch == 0:
                        batch.append(new_image_dict[sub_class][:self.batch_size])
                        temp.append(new_image_dict[sub_class][:self.batch_size])
                        new_image_dict[sub_class] = new_image_dict[sub_class][self.batch_size:]
                    else:
                        batch.append(new_image_dict[sub_class][:self.batch_size - len_of_batch])
                        temp.append(new_image_dict[sub_class][:self.batch_size - len_of_batch])
                        new_image_dict[sub_class] = new_image_dict[sub_class][self.batch_size - len_of_batch:]

            for i in temp:
                for j in i:
                    tempi.append(j[1])
            if len(tempi) == self.batch_size:
                total_batches.append(batch)
                count +=1
                batch = []
                tempi = []
                temp = []
            if count == (len(self.image_dict) // self.batch_size):
                finished = 1

        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def makeBatchesBinaryLabelled(self):
        binary_classification_batch = []
        for batch in self.dataset:
            if batch[1] != 0:
                binary_classification_batch.append((batch[0],1))
            else:
                binary_classification_batch.append((batch[0],0))
        self.dataset = binary_classification_batch

    def __getitem__(self, idx):
        # we use SequentialSampler together with SuperLabelTrainDataset,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]
        cls = batch_item[1]
        original_image = cv2.imread(batch_item[0])
        res_image = Image.fromarray(original_image)
        if cls == 0:
            return cls, self.transform(self.ensure_3dim(res_image))
        else:
            return cls, self.transform_query(self.ensure_3dim(res_image))

    def __len__(self):
        return len(self.dataset)
