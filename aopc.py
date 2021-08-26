"""
    This file incorporates the AOPC algorithm from Samek et. al 2017
"""

import utils
import torch

import matplotlib.pyplot as plt
import numpy as np
import copy

class AOPC:
    def __init__(self, image_data, image_idx, model, standard_model, perturbation_steps=100, perturbation_area=(3,9,9)):
        """
        image_data: images in equally shaped form to be evaluated. Contains data and classlabel
        image_data is not yet normalized for AlexNet use!
        model: neural network for calculating the relevance map and the prediction score
        perturbation_area: area to perturb around the relevance pixel, using channels
        also compares to random sequence

        using most probable class as label from the first non-perturbed image
        """
        self.image_data = image_data
        self.image_idx = image_idx #prediction idx of model
        self.len_images = image_data.shape[0]
        self.model = model
        self.standard_model = standard_model
        self.perturbation_steps = perturbation_steps
        self.perturbation_area = perturbation_area



    def _calc_r_sequences_sensitivity(self, norm="l2"):
        r_sequences = []
        for i in range(self.len_images):
            image = copy.deepcopy(self.image_data[i])
            image = image.unsqueeze(0)
            image.requires_grad_(True)
            pred = self.model(image)
            pred_score = pred[0][self.image_idx[i]]
            self.model.zero_grad()
            pred_score.backward()

            r_map = copy.deepcopy(self.model.gradients)
            if norm=="l2":
                r_map = torch.linalg.norm(r_map,ord=None, dim=1)
            elif norm=="l_inf":
                r_map = torch.linalg.norm(r_map,ord=np.inf, dim=1)

            r_map = r_map.squeeze()
            ordered_r_sequence = utils.relevance_map_to_ordered_sequence(r_map)
            r_sequences.append(ordered_r_sequence)

            del r_map
            del image

        print("Finished _calc_r_sequences_sensitivity() with norm: ", norm)
        return r_sequences

    def _calc_r_sequences_deconv(self, norm="l2"):
        r_sequences = []
        for i in range(self.len_images):
            image = copy.deepcopy(self.image_data[i])
            image = image.unsqueeze(0)

            pred = self.model(image)
            pred_score = pred[0][self.image_idx[i]]
            self.model.zero_grad()
            pred_score.backward()

            r_map = self.model.calc_heat()
            if norm=="l2":
                r_map = torch.linalg.norm(r_map,ord=None, dim=1)
            elif norm=="l_inf":
                r_map = torch.linalg.norm(r_map,ord=np.inf, dim=1)
            #r_map = r_map.sum(axis=1)
            #(x,y) i.e. (224,224)
            r_map = r_map.squeeze()
            ordered_r_sequence = utils.relevance_map_to_ordered_sequence(r_map)
            r_sequences.append(ordered_r_sequence)

            del r_map
            del image

        print("Finished _calc_r_sequences_deconv() with norm: ", norm)
        return r_sequences

        #lrp_rule:  epsilon0
        #       epsilon0_01
        #    epsilon100
    def _calc_r_sequences_lrp(self, lrp_rule="epsilon0_01"):
        #iterate over the image data and calculate the r-sequence
        r_sequences = []
        for i in range(self.len_images):
            image = copy.deepcopy(self.image_data[i])
            image = image.unsqueeze(0)
            #image = utils.normalize_image(self.image_data[i]) #normalized minibatch of size 1
            #print(image)
            #utils.show_image(image[0])
            image.requires_grad_(True)
            image.grad = None
            #TODO: Use absolute or softmax pred score #using single image
            pred = self.model(image, explain=True, rule=lrp_rule)
            self.model.zero_grad()
            #pred = torch.nn.functional.softmax(pred, dim=1)
            label = self.image_idx[i]
            pred_score = pred[0,label]
            #print("prediction score of label: ",pred_score)
            pred_score.backward()
            r_map = image.grad

            #summing up channel values
            r_map = r_map.sum(axis=1)
            r_map = r_map.squeeze()

            ordered_r_sequence = utils.relevance_map_to_ordered_sequence(r_map)
            r_sequences.append(ordered_r_sequence)

            #clean up
            del r_map
            del image

        print("Finished _calc_r_sequences() with rule: ",lrp_rule)
        return r_sequences



    def _recursive_MoRF(self, r_sequences):
        """
        create @perturbation_steps images recursively and then calculate
        the prediction for each additionally perturbed image (use batch for calculation)
        """
        aopc_data = torch.zeros((self.len_images,self.perturbation_steps+1))
        for i in range(self.len_images):
            perturbed_areas = torch.zeros((self.image_data.shape[1],self.image_data.shape[2],self.image_data.shape[3]))
            image_perturbations_tensor = torch.zeros((self.perturbation_steps+1,self.image_data.shape[1],self.image_data.shape[2],self.image_data.shape[3]))
            perturbed_image = copy.deepcopy(self.image_data[i])
            image_perturbations_tensor[0]=perturbed_image #original image

            i_p = 0
            while i_p < self.perturbation_steps:
                r_seq = r_sequences[i][i_p]
                #check if new perturbation would overlap with previous perturbation steps (Samek et. al "nonoverlapping regions")
                if utils.area_value(perturbed_areas,r_seq,self.perturbation_area) > 0.0:
                    #keep index and pop element from list
                    r_sequences[i].pop(i_p)
                    continue

                perturbed_image = utils.MoRF(perturbed_image,r_seq,self.perturbation_area, normalized=True)
                 #use perturb_value one here, in order to check the area value better (>0)
                perturbed_areas = utils.MoRF(perturbed_areas,r_seq,self.perturbation_area, perturb_value=torch.ones(self.perturbation_area))
                image_perturbations_tensor[i_p+1]= perturbed_image
                i_p += 1

            #utils.show_image(perturbed_image)
            del perturbed_areas
            del perturbed_image

            #(perturbations,3,224,224)

            #cuda
            if(torch.cuda.is_available() == True):
                image_perturbations_tensor = image_perturbations_tensor.to(device='cuda')

            #(perturbations,1000)
            #the order is mixed up....
            perturbed_prediction_scores = self.standard_model(image_perturbations_tensor)
            perturbed_prediction_scores = torch.nn.functional.softmax(perturbed_prediction_scores, dim=1) #Fucking hell. Took long enough to see, that dim=1 is needed...

            #use only prediction from the correct label
            label = self.image_idx[i]
            #(perturbations)
            perturbed_prediction_scores = perturbed_prediction_scores[:,label]

            #calculate AOPC

            # perturbed_prediction_scores = torch.abs(perturbed_prediction_scores - perturbed_prediction_scores[0])
            # aopc_data[i]=perturbed_prediction_scores.cpu().data

            a = self._calc_aopc(perturbed_prediction_scores.data)
            aopc_data[i] = a

            #delete variables explicitly, to avoid memory leak/ too much memory occupation in short time
            del perturbed_prediction_scores
            del label
            del image_perturbations_tensor
        return aopc_data


    def _calc_aopc(self,prediction_scores):
        prediction_scores = prediction_scores.cpu()
        L = prediction_scores.shape[0]
        aopc = torch.zeros((L))
        aopc[0] = torch.Tensor([0.0])
        for i in range(1, L):
            a = 1 / (i+1)
            ones = torch.ones((i+1))
            morf_0 = ones*prediction_scores[0]
            a = a* torch.sum(torch.sub(morf_0, prediction_scores[0:i+1]))
            #a = torch.max(torch.Tensor([0.0]),a)
            aopc[i]=a

        return aopc

        #rule:  epsilon0
        #       epsilon0_01
        #       epsilon100
        #       deconv
        #       sensitivity
    def get_aopc_curve(self, rule="epsilon0_01", norm="l2"):
        """
        average over all images.
        compare to random sequence (_recursive_MoRF_random)
        """
        r_sequences = None
        if rule == "deconv":
            r_sequences = self._calc_r_sequences_deconv(norm)
        elif rule == "sensitivity":
            r_sequences = self._calc_r_sequences_sensitivity(norm)
        else:
            r_sequences = self._calc_r_sequences_lrp(rule)

        if r_sequences == None:
            raise Exception('aopc.get_aopc_curve with rule: ',rule,'and norm: ', norm, ' could not calculate any r_sequences!')

        aopc_data = self._recursive_MoRF(r_sequences)

        #return aopc_data
        print("Finished aopc data with rule: ", rule)
        return torch.mean(aopc_data,dim=0,keepdim=True)

    def compare_to_random(self):
        """
        do everything again, but with random relevance maps?
        """
        random_r_sequences = []
        for i in range(self.len_images):
            random_r_sequences.append(utils.create_random_sorted_r_seq((self.image_data.shape[2],self.image_data.shape[3])))
        aopc_data_random = self._recursive_MoRF(random_r_sequences)
        
        #return aopc_data_random
        return torch.mean(aopc_data_random,dim=0,keepdim=True)
