import numpy
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.2, aug)

from PIL import  Image, ImageFilter


aug_transform = iaa.SomeOf((0,2),[
# iaa.Fliplr(0.5),
sometimes(iaa.LinearContrast((0.75,1.5))),
iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
# Same as sharpen, but for an embossing effect.
iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
sometimes(iaa.OneOf([
    iaa.EdgeDetect(alpha=(0, 0.7)),
    iaa.DirectedEdgeDetect(
        alpha=(0, 0.7), direction=(0.0, 1.0)
    ),
])),
iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
iaa.OneOf([
    iaa.Dropout((0.01, 0.02), per_channel=0.5),
    iaa.CoarseDropout(
        (0.02, 0.15), size_percent=(0.005, 0.008)),
]),
iaa.Add((-10, 10), per_channel=0.5),
iaa.Multiply((0.5, 1.5), per_channel=0.2),
iaa.Sometimes(0.1,iaa.OneOf([
    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
    iaa.PiecewiseAffine(scale=(0.005, 0.02))]))

# iaa.CenterCropToFixedSize()
],random_order=True)
class ImgAug(object):
    def __init__(self):
        self.aug = aug_transform
    def __call__(self,img):
        array_img = self.aug(image = numpy.array(img))
        # return Image.fromarray(array_img)
        return array_img
    def __repr__(self):
        return self.__class__.__name__