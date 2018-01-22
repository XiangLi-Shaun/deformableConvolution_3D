
# coding: utf-8


import numpy as np
import tensorflow as tf

class DCN(object):
    
    """
    Init func
    Only two params：
    input_shape：input feature map shape, a 1*5 list [N,H,W,D,C]
    kernel_size：a 1*3 list
    """
    def __init__(self, input_shape, kernel_size):
        
        # 定义deform_conv的kernel size
        self.kernel_size = kernel_size
        # self.num_points = kernel_size[0]*kernel_size[1]
        self.num_points = kernel_size[0]*kernel_size[1]*kernel_size[2]
        
        # define feature map shape
        self.num_batch = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.depth = input_shape[3] #for 3D
        # self.num_channels = input_shape[3]
        self.num_channels = input_shape[4]
        
        self.extend_scope = 3.0
    
    
    """
    _coordinate_map(self, offset_field) will generate [3W,3H,3D] coordinate map
    input：offset field: [N,H,W,D,3*3*3*3]
    output：[N,3W,3H,3D] coordinate map
    """
    def _coordinate_map_3D(self, offset_field, name):
        with tf.variable_scope(name+"/_coordinate_map"):
            
            # offset
            # x_offset, y_offset = tf.split(tf.reshape(offset_field, [self.num_batch, self.height, self.width, 2, self.num_points]),2,3)
            x_offset, y_offset, z_offset = tf.split(tf.reshape(offset_field, [self.num_batch, self.height, self.width, self.depth, 3, self.num_points]),3,4)
            x_offset = tf.squeeze(x_offset) #[N,H,W,D,3*3*3]
            y_offset = tf.squeeze(y_offset)
            z_offset = tf.squeeze(z_offset)

            # center coordinate
            # x_center = tf.reshape(tf.tile(tf.range(self.width),[self.height]),[self.height*self.width,-1])
            x_center = tf.tile(tf.range(self.width),[self.height*self.depth])
            x_center = tf.transpose(tf.reshape(x_center, [self.height,self.depth,self.width]),[0,2,1])
            x_center = tf.reshape(x_center,[self.height*self.width*self.depth,-1])
            x_center = tf.tile(x_center,[1,self.num_points])
            x_center = tf.reshape(x_center,[self.height,self.width,self.depth,self.num_points])
            x_center = tf.tile(tf.expand_dims(x_center, 0), [self.num_batch,1,1,1,1])

            # y_center = tf.tile(tf.range(self.height),[self.width])
            y_center = tf.tile(tf.range(self.height),[self.width*self.depth])
            y_center = tf.transpose(tf.reshape(y_center, [self.width,self.depth,self.height]),[2,0,1])
            y_center = tf.reshape(y_center,[self.height*self.width*self.depth,-1])
            y_center = tf.tile(y_center,[1,self.num_points])
            y_center = tf.reshape(y_center,[self.height,self.width,self.depth,self.num_points])
            y_center = tf.tile(tf.expand_dims(y_center, 0), [self.num_batch,1,1,1,1])

            # z_center
            z_center = tf.tile(tf.range(self.depth),[self.height*self.width])
            z_center = tf.transpose(tf.reshape(z_center, [self.height,self.width,self.depth]),[0,1,2]) #actually no transpose
            z_center = tf.reshape(z_center,[self.height*self.width*self.depth,-1])
            z_center = tf.tile(z_center,[1,self.num_points])
            z_center = tf.reshape(z_center,[self.height,self.width,self.depth,self.num_points])
            z_center = tf.tile(tf.expand_dims(z_center, 0), [self.num_batch,1,1,1,1])
        
            x_center = tf.cast(x_center,"float32")
            y_center = tf.cast(y_center,"float32")
            z_center = tf.cast(z_center,"float32") #[N,H,W,D,3*3*3]

            # regular grid
            x = tf.linspace(float(-(self.kernel_size[0]-1)/2), float((self.kernel_size[0]-1)/2), self.kernel_size[0])
            y = tf.linspace(float(-(self.kernel_size[1]-1)/2), float((self.kernel_size[1]-1)/2), self.kernel_size[1])
            z = tf.linspace(float(-(self.kernel_size[2]-1)/2), float((self.kernel_size[2]-1)/2), self.kernel_size[2])
            
            x,y,z = tf.meshgrid(x,y,z)
            x_spread = tf.transpose(tf.reshape(x,(-1,1)))
            y_spread = tf.transpose(tf.reshape(y,(-1,1)))
            z_spread = tf.transpose(tf.reshape(z,(-1,1)))

            x_grid = tf.tile(x_spread,[1,self.height*self.width*self.depth])
            x_grid = tf.reshape(x_grid, [self.height, self.width, self.depth, self.num_points])
            y_grid = tf.tile(y_spread,[1,self.height*self.width*self.depth])
            y_grid = tf.reshape(y_grid, [self.height, self.width, self.depth, self.num_points])
            z_grid = tf.tile(z_spread,[1,self.height*self.width*self.depth])
            z_grid = tf.reshape(z_grid, [self.height, self.width, self.depth, self.num_points])
            x_grid = tf.tile(tf.expand_dims(x_grid, 0), [self.num_batch,1,1,1,1])
            y_grid = tf.tile(tf.expand_dims(y_grid, 0), [self.num_batch,1,1,1,1])
            z_grid = tf.tile(tf.expand_dims(z_grid, 0), [self.num_batch,1,1,1,1]) #[N,H,W,D,3*3*3]


            # calculate X,Y,Z
            x = tf.add_n([x_center, x_grid, tf.multiply(self.extend_scope, x_offset)])
            y = tf.add_n([y_center, y_grid, tf.multiply(self.extend_scope, y_offset)])
            z = tf.add_n([z_center, z_grid, tf.multiply(self.extend_scope, z_offset)]) #[N,H,W,D,3*3*3]

            # uncomment this to be normal CNN layer
            # x = tf.add_n([x_center, x_grid, tf.multiply(0.0, x_offset)])
            # y = tf.add_n([y_center, y_grid, tf.multiply(0.0, y_offset)])
            # z = tf.add_n([z_center, z_grid, tf.multiply(0.0, z_offset)]) #[N,H,W,D,3*3*3]

            # reshape N*H*W*D*num_points to N*3H*3W*3D
            x_new = tf.reshape(x,[self.num_batch,self.height,self.width,self.depth,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]])
            x_new = tf.reshape(tf.transpose(x_new,[0,1,4,2,5,3,6]),[self.num_batch,self.kernel_size[0]*self.height,self.kernel_size[1]*self.width,self.kernel_size[2]*self.depth])

            
            y_new = tf.reshape(y,[self.num_batch,self.height,self.width,self.depth,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]])
            y_new = tf.reshape(tf.transpose(y_new,[0,1,4,2,5,3,6]),[self.num_batch,self.kernel_size[0]*self.height,self.kernel_size[1]*self.width,self.kernel_size[2]*self.depth])


            z_new = tf.reshape(z,[self.num_batch,self.height,self.width,self.depth,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]])
            z_new = tf.reshape(tf.transpose(z_new,[0,1,4,2,5,3,6]),[self.num_batch,self.kernel_size[0]*self.height,self.kernel_size[1]*self.width,self.kernel_size[2]*self.depth])

            return x_new, y_new, z_new
    
    """
    _bilinear_interpolate(self, input_feature, coordinate_map) will bilinearly interpolate
    input：input feature map [N,W,H,D,C]；coordinate map [N,3W,3H,3D]
    output：[3W,3H,C1] deformed feature map
    """
    def _bilinear_interpolate_3D(self, input_feature, x, y, z, name):
        with tf.variable_scope(name+"/_bilinear_interpolate"):
        
            # flatten to 1D
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])
            z = tf.reshape(z, [-1]) #[N*3W*3H*3D]


            # data type convertion
            x = tf.cast(x, "float32")
            y = tf.cast(y, "float32")
            z = tf.cast(z, "float32")
            zero = tf.zeros([], dtype="int32")
            max_x = tf.cast(self.width-1, "int32")
            max_y = tf.cast(self.height-1, "int32")
            max_z = tf.cast(self.depth-1, "int32")


            # find 8 grid locations
            x0 = tf.cast(tf.floor(x), "int32")
            x1 = x0+1
            y0 = tf.cast(tf.floor(y), "int32")
            y1 = y0+1
            z0 = tf.cast(tf.floor(z), "int32")
            z1 = z0+1

            # clip out coordinates exceeding feature map volume以外的点
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            z0 = tf.clip_by_value(z0, zero, max_z)
            z1 = tf.clip_by_value(z1, zero, max_z) #[N*3H*3W*3D]

            # convert input_feature and coordinate X, Y to 3D，for gathering
            input_feature_flat = tf.reshape(input_feature, tf.stack([-1, self.num_channels])) #[N*H*W*D,C]

            dimension_3 = self.depth
            dimension_2 = self.depth*self.width
            dimension_1 = self.depth*self.width*self.height
            base = tf.range(self.num_batch)*dimension_1
            repeat = tf.transpose(tf.expand_dims(tf.ones(shape=(tf.stack([self.num_points*self.height*self.width*self.depth,]))),1),[1,0]) #[1,H*W*D*27]
            repeat = tf.cast(repeat,"int32") #[1,H*W*D*27]
            base = tf.matmul(tf.reshape(base,(-1,1)),repeat) # [N,1] * [1,H*W*D*27] ==> [N,H*W*D*27]
            base = tf.reshape(base, [-1]) #[H*W*D*27]
            base_x0 = base + x0*dimension_3
            base_x1 = base + x1*dimension_3
            base_y0 = base + y0*dimension_2
            base_y1 = base + y1*dimension_2

            #top rectangle of the neighbourhood volume
            index_a0 = base_y0 + base_x0 - base + z0
            index_b0 = base_y0 + base_x1 - base + z0
            index_c0 = base_y0 + base_x0 - base + z1
            index_d0 = base_y0 + base_x1 - base + z1 #[N*3H*3W*3D]

            #bottom rectangle of the neighbourhood volume
            index_a1 = base_y1 + base_x0 - base + z0
            index_b1 = base_y1 + base_x1 - base + z0
            index_c1 = base_y1 + base_x0 - base + z1
            index_d1 = base_y1 + base_x1 - base + z1 #[N*3H*3W*3D]

            # get 8 grid values  ([N*H*W*D,C], [N*H*W*D*27])
            value_a0 = tf.gather(input_feature_flat, index_a0)
            value_b0 = tf.gather(input_feature_flat, index_b0)
            value_c0 = tf.gather(input_feature_flat, index_c0)
            value_d0 = tf.gather(input_feature_flat, index_d0) #[N*3H*3W*3D, C]
            value_a1 = tf.gather(input_feature_flat, index_a1)
            value_b1 = tf.gather(input_feature_flat, index_b1)
            value_c1 = tf.gather(input_feature_flat, index_c1)
            value_d1 = tf.gather(input_feature_flat, index_d1) #[N*3H*3W*3D, C]

            # calculate 8 volumes : need to be diagonal volume for corresponding point
            x0_float = tf.cast(x0, "float32")
            x1_float = tf.cast(x1, "float32")
            y0_float = tf.cast(y0, "float32")
            y1_float = tf.cast(y1, "float32")
            z0_float = tf.cast(z0, "float32")
            z1_float = tf.cast(z1, "float32")
            vol_a0 = tf.expand_dims(((x1_float-x)*(y1_float-y)*(z1_float-z)),1)
            vol_b0 = tf.expand_dims(((x-x0_float)*(y1_float-y)*(z1_float-z)),1)
            vol_c0 = tf.expand_dims(((x1_float-x)*(y1_float-y)*(z-z0_float)),1)
            vol_d0 = tf.expand_dims(((x-x0_float)*(y1_float-y)*(z-z0_float)),1) 
            vol_a1 = tf.expand_dims(((x1_float-x)*(y-y0_float)*(z1_float-z)),1)
            vol_b1 = tf.expand_dims(((x-x0_float)*(y-y0_float)*(z1_float-z)),1)
            vol_c1 = tf.expand_dims(((x1_float-x)*(y-y0_float)*(z-z0_float)),1)
            vol_d1 = tf.expand_dims(((x-x0_float)*(y-y0_float)*(z-z0_float)),1) #[N*3H*3W*3D, 1]

            ########################
            outputs = tf.add_n([value_a0*vol_a0, value_b0*vol_b0, value_c0*vol_c0, value_d0*vol_d0,value_a1*vol_a1, value_b1*vol_b1, value_c1*vol_c1, value_d1*vol_d1])
            outputs = tf.reshape(outputs, [self.num_batch, self.kernel_size[0]*self.height, self.kernel_size[1]*self.width, self.kernel_size[2]*self.depth, self.num_channels])

        
            return outputs #[N,3W,3H,3D,C]


       
    
    """
    deform_conv(self, inputs) operates deformable convolution
    input：input feature map
    output：output feature map
    """
    def deform_conv(self, inputs, offset, name, **kwargs):
        with tf.variable_scope(name+"/DeformedFeature"):
            x, y, z = self._coordinate_map_3D(offset, name)
            deformed_feature = self._bilinear_interpolate_3D(inputs, x, y, z, name)
            return deformed_feature



