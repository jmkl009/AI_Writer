"""
Story generation
"""
from six.moves import cPickle as pkl
import numpy
import copy
import sys
import skimage.transform

import skipthoughts
import decoder
"""
Configuration for the generate module
"""

#-----------------------------------------------------------------------------#
# Flags for running on CPU
#-----------------------------------------------------------------------------#
FLAG_CPU_MODE = True

#-----------------------------------------------------------------------------#
# Paths to models and biases
#-----------------------------------------------------------------------------#
paths = dict()

# Skip-thoughts
paths['skmodels'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/skmodels/'
paths['sktables'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/skmodels/'

# Decoder
paths['decmodel'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/omance.npz'
paths['dictionary'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/romance_dictionary.pkl'

# Image-sentence embedding
paths['vsemodel'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/coco_embedding.npz'

# VGG-19 convnet
paths['vgg'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller//vgg19.pkl'
paths['pycaffe'] = '/u/yukun/Projects/caffe-run/python'
paths['vgg_proto_caffe'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/VGG_ILSVRC_19_layers_deploy.prototxt'
paths['vgg_model_caffe'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/VGG_ILSVRC_19_layers.caffemodel'


# COCO training captions
paths['captions'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/coco_train_caps.txt'

# Biases
paths['negbias'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/caption_style.npy'
paths['posbias'] = '/Users/wangjingjin/PycharmProjects/AI_Writer/neural_storyteller/romance_style.npy'