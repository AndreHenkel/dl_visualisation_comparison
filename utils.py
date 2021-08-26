#this utils class has multiple functions to ease up development etc.
#also especially for Samek et. al

from PIL import Image
from torchvision import transforms
import numpy as np

import torch
from dataclasses import dataclass
import operator

import matplotlib.pyplot as plt

@dataclass
class R_pix:
    def __init__(self,x: int, y: int, r_value: float):
        self.x = x
        self.y = y
        self.r_value = r_value

    def __repr__(self):
        return f"x={self.x},y={self.y},r={self.r_value}" #Python >= 3.6
    def __str__(self):
        return f"x={self.x},y={self.y},r={self.r_value}" #Python >= 3.6



def area_value(image: torch.tensor, r_pix, area_size=(3,9,9)):
    """
    calculates the value of this area.
    This is handy to see if a torch.zeros image was perturbed in this area
    """
    x_start = r_pix.x - ((area_size[1] - 1)/2)
    y_start = r_pix.y - ((area_size[2] - 1)/2)

    #bound value
    x_start = int(max(0,min(image.shape[1]-area_size[1],x_start)))
    y_start = int(max(0,min(image.shape[2]-area_size[2],y_start)))

    val = torch.sum(image[:,x_start:x_start+area_size[1],y_start:y_start+area_size[2]])
    return val



def MoRF(image: torch.Tensor, r_pix: R_pix, area_size=(3,9,9), normalized=True, perturb_value=None):
    """
    Expects image to be in shape of [channels,width,height] and with values from 0 - 1
    area_size values should be uneven, andthe first value are the channels
    Morfs an given @image at the position given by @r_pix with randomly sampled values in the area of size @area_size
    Distinguishes between between channels and
    range_mul multiples the range [-0.5,0.5]
    """
    # if torch.min(image) < 0.0 or torch.max(image) > 1.0:
    #     raise Exception('utils.MoRF @image values are out of bound [0,1]', torch.min(image), torch.max(image))

    area_morf_rnd = torch.rand(area_size) # [0,1]

    if normalized==True:
        area_morf_rnd = (torch.rand(area_size) - 0.5) * 1 # with 5 -> in range from -2.5 to 2.5, which is approx. the range in the normalized image

    if perturb_value != None:
        area_morf_rnd = perturb_value


    x_start = r_pix.x - ((area_size[1] - 1)/2)
    y_start = r_pix.y - ((area_size[2] - 1)/2)

    #bound value
    x_start = int(max(0,min(image.shape[1]-area_size[1],x_start)))
    y_start = int(max(0,min(image.shape[2]-area_size[2],y_start)))

    image[:,x_start:x_start+area_size[1],y_start:y_start+area_size[2]] = area_morf_rnd

    return image

def relative_graph(base, to,scaling=1000):
    """
    returns relative tensor of the same shape
    usually relative is calculated as percentage.
    But since a percentage wise graph would lead to massive values in the first couple perturbation steps,
    this approach was used instead for a smoother graph
    """
    #rejected percentage wise relative graph
    #ret = torch.mul(torch.div(torch.sub(to,base),base),100)

    #diff scaling approach -> smoother graph
    ret = torch.sub(to,base) * scaling
    ret[ret != ret] = 0#set nan (first element) to 0
    return ret

#from TorchLRP
def project(X, output_range=(0, 1)):
    absmax   = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
    X       /= absmax + (absmax == 0).astype(float)
    X        = (X+1) / 2. # range [0, 1]
    X        = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
    return X

def create_random_sorted_r_seq(r_map_size=(224,224)):
    """
    r_map_size: (224,224) default for pretrained alexnet
    Returns: a relevance map of the specified size with random values reaching from 0 - 255
        as sorted sequence
    """

    rnd_r_map = torch.rand(size=r_map_size)
    sorted_r_seq = relevance_map_to_ordered_sequence(rnd_r_map)
    return sorted_r_seq


def relevance_map_to_ordered_sequence(r_map: torch.Tensor, area=(3,9,9), max_steps=10000):
    """
    r_map of size mxn
    Sorts a relevance map into an ordered sequence by relevance value from high to low
    in max_steps one has to consider, that overlapping r_pix positions are skipped
    """


    # #pytorch approach
    #depending on the perturbation steps and the area, it is possible, that the max_steps are reached and an error occurs,
    #that's why I went back to the "old approach" below, which calculates all R_pix values. i.e. perturbation_steps = 100 * area = 9x9 -> 8100 max R_pix needed,
    #due to overlapping
    # min_value = torch.min(r_map)
    # r_sequence = []
    # for i in range(max_steps):
    #     max_pos = (r_map==torch.max(r_map)).nonzero()[0] #take first appearance of the max
    #     r_pix = R_pix(max_pos[0], max_pos[1], r_map[max_pos[0],max_pos[1]].cpu().data.item())
    #     r_sequence.append(r_pix)
    #     #reset value to small value
    #     r_map[max_pos[0],max_pos[1]] = min_value
    #
    # return r_sequence

    # old approach
    r_map = r_map.cpu()
    r_sequence = []
    for x in range(r_map.shape[0]):
        for y in range(r_map.shape[1]):
            r_sequence.append(R_pix(x,y,r_map[x][y].data.item()))

    sorted_r_map = sorted(r_sequence, key=operator.attrgetter('r_value'),reverse=True)

    return sorted_r_map


def show_image(image_tensor: torch.Tensor):
    plt.imshow(image_tensor.permute(1,2,0).cpu())
    plt.show()

def load_image(filename):
    input_image = Image.open(filename)
    resize = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = resize(input_image)
    return image

#normalize image
def normalize_image(image: torch.Tensor):
    """
      Normalizes the image taken from the given filename, as expected for all pretrained pytorch models from the torch.hub.
      filename: location of file to be loaded
      return: minibatch of size 1 of normalized image as tensor(on cpu)
    """
    normalize = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    normalized_image = normalize(image)
    image_batch = normalized_image.unsqueeze(0) # create a mini-batch as expected by the model
    return image_batch


#load class labels for alexnet
def determine_topk_classes(probabilities,k = 5):
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    topk_prob, topk_catid = torch.topk(probabilities, k)
    print("-----Top",k,"Results------")
    for i in range(topk_prob.size(0)):
        print(topk_catid[i].item(), categories[topk_catid[i]], topk_prob[i].item())
    print("---------------------------")


def calculate_linear_lrp(in_layer_activations: torch.Tensor, weights: torch.Tensor, out_layer_relevance: torch.Tensor, epsilon: torch.Tensor = torch.Tensor([0.0])) -> torch.Tensor:
    """
    @in_layer_activations: the activations from each neuron from the "previous" layer
    @weights: the weights between the previous and the latter layer. torch.Tensor of shape [out_layer_relevance.shape, in_layer_activations.shape]
    @out_layer_relevance: The relevance values for each neuron from the latter layer.
    @epsilon: The epsilon value for the LRP_epsilon algorithm
    @return: The relevance values for each neuron from the @in_layer_activations
    """

    #First implementation with very slow python loop, to be sure, that the right things were calculated

    R = torch.zeros(in_layer_activations.shape)

    jk_act = torch.mul(in_layer_activations,weights)
    print(jk_act.shape)

    #r-loop
    for r_j in range(0, R.shape[0]):
        #k-loop (latter)
        relevance = torch.zeros((out_layer_relevance.shape))
        for k in range(0,out_layer_relevance.shape[0]):
            relevance[k] = in_layer_activations[r_j] *  weights[k,r_j]
            lower_part = epsilon + torch.sum(jk_act[k:])
            relevance[k] = relevance[k] / lower_part
            relevance[k] = relevance[k] *out_layer_relevance[r_j]

        R[r_j] = torch.sum(relevance[k])

    return R

#old approach before using faster lrp_framework
def calculate_linear_lrp_fast(in_layer_activations: torch.Tensor, weights: torch.Tensor, out_layer_relevance: torch.Tensor, epsilon: torch.Tensor = torch.Tensor([0.0])) -> torch.Tensor:
    """
    @in_layer_activations: the activations from each neuron from the "previous" layer
    @weights: the weights between the previous and the latter layer. torch.Tensor of shape [out_layer_relevance.shape, in_layer_activations.shape]
    @out_layer_relevance: The relevance values for each neuron from the latter layer.
    @epsilon: The epsilon value for the LRP_epsilon algorithm
    @return: The relevance values for each neuron from the @in_layer_activations
    """

    R = torch.nn.ReLU()(torch.mul(weights, in_layer_activations))
    R = torch.mul(R.T, out_layer_relevance)
    denominator = torch.add(torch.sum(R,dim=0),epsilon)
    R = torch.divide(R,denominator)
    R = torch.sum(R, dim=1)
    return R
