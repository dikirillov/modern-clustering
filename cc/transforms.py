import torchvision
import torchvision.transforms.v2


class CCTransforms:
    def __init__(self, size, mean=None, std=None):
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                                                  p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.v2.GaussianBlur(kernel_size=23),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=mean, std=std)
            ]
        )

    def __call__(self, x):
        return self.transforms(x), self.transforms(x)
