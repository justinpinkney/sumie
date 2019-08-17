__Out of date__

# Feature visualisation

Sumie ls a library that provides ways of producing visualisation through optimisation for a particular response form a neural network. The classic example of this is to produce an image that maximally activates a particular part of a network, as a way of visualising what features it has learnt.

To get a 'nice' looking image this way isn't as simple as it seems, and lots of people have figure out plenty of tricks to get this to work well. But sumie wants to make this easy.

First we need a pretrained neural network that we want to visualise the features for. Happily torchvision comes with plenty of pretrained models.

```python
import sumie
from torchvision import models

net = models.vgg16(pretrained=True)
```

Once we've loaded a model we need to pick a part of it to visualise. We reference any torch sub module in the network, a good starting point is to look at the last part of the network which is used for classification. These should correspond to the different classes the network has been trained to recognise in imagenet.

```python
target_layer = net.classifier[6]
im = sumie.visualise(net.classifier[6], 0)
```

at a lower level:

```python
image = sumie.Image(224, 'fft',
                    decorrelate=True,
                    limit='sigmoid',
                    transforms=transforms)
target = sumie.objectives.channel(net.classifier[6], 0)
output_image = sumie.visualise(image, target)
# Also specify, optimiser, iterations
```

fft/rgb/cppn
decorelate/not
sigmoid/clip
transforms (many)
normalisation
