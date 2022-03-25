from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        height = (int)((input_size[1] + 2*self.padding - self.kernel_size) / (self.stride) + 1)
        width = (int)((input_size[2] + 2*self.padding - self.kernel_size) / (self.stride) + 1)
        output_shape = [input_size[0], height, width, self.number_filters ]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        #pad the input image according to self.padding (see np.pad)
        padded_img = np.pad(img,((0, 0), (self.padding, self.padding), (self.padding, self.padding),(0,0)))
        _, padded_height, padded_width, _ = padded_img.shape

        #iterate over output dimensions, moving by self.stride to create the output
        filters = self.params[self.w_name]
        output = np.zeros((output_shape))
        
        for i in range(output_shape[1]):
            for j in range(output_shape[2]): 
                row_start= i*self.stride
                row_end= row_start + self.kernel_size
                col_start= j*self.stride
                col_end = col_start + self.kernel_size             
                output[:, i, j, :] = np.sum(padded_img[:,row_start:row_end,col_start:col_end,:,np.newaxis]*filters[np.newaxis,:,:,:],axis=(1, 2, 3))
        
        output = output+self.params[self.b_name]
        self.meta = padded_img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        filters = self.params[self.w_name]
        self.grads[self.w_name] = np.zeros((filters.shape))

        dX = np.zeros((img.shape))
        _ , input_height, input_width, _ = img.shape
        _ , output_height, output_width, _ = dprev.shape
        for i in range(output_height):
            for j in range(output_width):
                row_start = i * self.stride
                row_end = row_start + self.kernel_size
                col_start = j * self.stride
                col_end = col_start + self.kernel_size
                dX[:,row_start:row_end,col_start:col_end,:] += np.sum(filters[np.newaxis,:,:,:,:]*dprev[:,i:i+1,j:j+1,np.newaxis,:],axis=4)
                self.grads[self.w_name] += np.sum(img[:,row_start:row_end,col_start:col_end,:,np.newaxis]*dprev[:,i:i+1,j:j+1,np.newaxis,:],axis=0)

        self.grads[self.b_name] = np.sum(dprev,axis=(0,1,2))
        if(self.padding!=0):
            dX = dX[:,self.padding:-self.padding,self.padding:-self.padding,:]
        dimg = dX

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        self.meta = img
        _ , input_height, input_width, _ = img.shape
        output_height = (int)((input_height - self.pool_size)/(self.stride) + 1)
        output_width = (int)((input_width - self.pool_size)/(self.stride) + 1)
        
        output = np.zeros((img.shape[0],output_height,output_width,img.shape[3]))
        
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                row_start = i * self.stride
                row_end = row_start + self.pool_size
                col_start = j * self.stride
                col_end = col_start + self.pool_size
                img_prev_slice = img[:,row_start:row_end, col_start:col_end, :]
                output[:, i, j, :] = np.max(img_prev_slice,axis=(1,2))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################

        for i in range(h_out):
            for j in range(w_out):
                row_start = i * self.stride
                row_end = row_start + self.pool_size
                col_start = j * self.stride
                col_end = col_start + self.pool_size

                img_prev_slice = img[:,row_start:row_end,col_start:col_end,:]
                # Calculate mask
                img_prev_cpy = img_prev_slice
                mask = np.zeros_like(img_prev_slice)
                nC,h,w,ch = img_prev_slice.shape
                img_prev_cpy = img_prev_cpy.reshape(nC,h*w,ch)
                idx = np.argmax(img_prev_cpy, axis=1)
                n_idx,ch_idx = np.indices((nC,ch))
                mask.reshape(nC,h*w,ch)[n_idx, idx, ch_idx] = 1
                
                dimg[:, row_start:row_end,col_start:col_end,:] += dprev[:,i:i+1,j:j+1,:]*mask

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
