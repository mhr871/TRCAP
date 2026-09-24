from torchvision import transforms

from .tasviret import TasvirEtTrain, TasvirEtTest


def getDino2Transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )


def getTestTransforms(model_config: dict):
    return getDino2Transforms(model_config.get("image_size", 224))


def getTrainDataset(dataset_root: str, json_path: str, model_config: dict):
    transform = getTestTransforms(model_config)
    return TasvirEtTrain(dataset_root=dataset_root, json_path=json_path, transforms=transform)


def getTestDataset(dataset_root: str, json_path: str, model_config: dict):
    transform = getTestTransforms(model_config)
    return TasvirEtTest(dataset_root=dataset_root, json_path=json_path, transforms=transform)
