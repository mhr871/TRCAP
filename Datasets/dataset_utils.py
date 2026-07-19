
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .coco import COCOKarpathyTrain, COCOKarpathyTest
from .flickr import FlickrTrain, FlickrTest
from .tasviret import TasvirEtTrain, TasvirEtTest
from Model import clip
from transform.randaugment import RandomAugment


def getTrainTransform():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToPILImage()
    ])
    return transform_train


def getDino2Transforms(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )


def getTestTransforms(vision_model=None, model_config=None):
    if model_config is not None and "dino2" in model_config:
        return getDino2Transforms(model_config.get("image_size", 224))

    if vision_model is None:
        _, preprocess = clip.load("ViT-B/32", jit=False)
    else:
        _, preprocess = clip.load(vision_model, jit=False)
    return preprocess


def getTrainDataset(dataset_name, dataset_root, train_json_path, vision_model=None, model_config=None):
    test_transforms = getTestTransforms(vision_model, model_config)
    # train_transforms = torchvision.transforms.Compose([getTrainTransform(),
    #                                                    test_transforms])
    train_transforms = test_transforms
    if dataset_name.lower() == 'coco-karphaty':
        train_dataset = COCOKarpathyTrain(dataset_root=dataset_root,
                                          json_path=train_json_path,
                                          tokenizer=None,
                                          transforms=train_transforms)
    elif dataset_name.lower() == 'tasvir-et':
        train_dataset = TasvirEtTrain(dataset_root=dataset_root,
                                      json_path=train_json_path,
                                      transforms=train_transforms)
    elif dataset_name.lower() == 'flickr30k':
        train_dataset = FlickrTrain(dataset_root=dataset_root,
                                    json_path=train_json_path,
                                    transforms=train_transforms)
    else:
        raise Exception(f"Unknown dataset : {dataset_name}")
    return train_dataset


def getTestDataset(dataset_name, dataset_root, test_json_path, vision_model=None, model_config=None):
    test_transforms = getTestTransforms(vision_model, model_config)
    if dataset_name.lower() == 'coco-karphaty':
        test_dataset = COCOKarpathyTest(dataset_root=dataset_root,
                                        json_path=test_json_path,
                                        transforms=test_transforms)
    elif dataset_name.lower() == 'tasvir-et':
        test_dataset = TasvirEtTest(dataset_root=dataset_root,
                                    json_path=test_json_path,
                                    transforms=test_transforms)
    elif dataset_name.lower() == 'flickr30k':
        test_dataset = FlickrTest(dataset_root=dataset_root,
                                  json_path=test_json_path,
                                  transforms=test_transforms)
    else:
        raise Exception(f"Unknown dataset : {dataset_name}")
    return test_dataset


def getCocoDataset(dataset_root, train_json_path, test_json_path, vision_model=None, model_config=None):
    test_transforms = getTestTransforms(vision_model, model_config)
    train_dataset = COCOKarpathyTrain(dataset_root=dataset_root,
                                      json_path=train_json_path,
                                      tokenizer=None,
                                      transforms=test_transforms)

    test_dataset = COCOKarpathyTest(dataset_root=dataset_root,
                                    json_path=test_json_path,
                                    transforms=test_transforms)
    return train_dataset, test_dataset


def getTasvirEtDataset(dataset_root, train_json_path, test_json_path, vision_model=None, model_config=None):
    test_transforms = getTestTransforms(vision_model, model_config)
    train_dataset = TasvirEtTrain(dataset_root=dataset_root,
                                  json_path=train_json_path,
                                  transforms=test_transforms)

    test_dataset = TasvirEtTest(dataset_root=dataset_root,
                                json_path=test_json_path,
                                transforms=test_transforms)
    return train_dataset, test_dataset
