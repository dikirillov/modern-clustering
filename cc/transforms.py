import torchvision
import torchvision.transforms.v2


class CCTransforms:
    def __init__(self, size, mean=None, std=None, is_test=False):
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                                                  p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.v2.GaussianBlur(kernel_size=23),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5)
            ]
        )

        if is_test:
            self.transforms = torchvision.transforms.Compose(
                [
                    T.ToTensor(),
                    T.Resize((32, 32)),
                    T.Normalize([0.5], [0.5])
                ]
            )
                
    def __call__(self, x):
        return self.transforms(x), self.transforms(x)
