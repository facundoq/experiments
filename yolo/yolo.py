
import torch.nn as nn
import torch
import numpy as np
class YOLO(nn.Module):

    def __init__(self,classes):
        super(YOLO,self).__init__()
        self.classes=classes


    def forward(self,x):
        return x



class BoundingBoxes:

    def __init__(self,input_grid_size,anchors):
        self.anchors=anchors
        h,w=input_grid_size
        self.centers = self.calculate_centers(h, w, len(anchors))
        self.centers = torch.autograd.Variable(torch.FloatTensor(self.centers))
        # replicate anchors for each cell
        flattened_input_size=h*w
        sizes = anchors * flattened_input_size
        self.sizes = torch.autograd.Variable(torch.FloatTensor(sizes))


    def calculate_centers(self,h,w,num_anchors):
        n=h*w*num_anchors
        deltaH,deltaW=1/h/2,1/w/2
        centers = [(i/h+deltaH, j/w+deltaW) for i in range(h) for j in range(w) for a in range(num_anchors)]
        assert (len(centers) == n)
        centers = np.array(centers)
        return centers

    def cuda(self):
        self.anchors=self.anchors.cuda()
        self.centers= self.centers.cuda()

# This layer converts CxHxW feature maps to HxWxB bounding box predictions
class DetectionLayer(nn.Module):
    def __init__(self,input_grid_size,anchors,classes,use_cuda=False):
        super(DetectionLayer,self).__init__()
        self.use_cuda=use_cuda
        self.anchors=anchors
        self.classes=classes
        self.input_grid_size=input_grid_size
        c, h, w = input_grid_size
        self.flattened_input_size = h * w
        coordinates = 4
        self.num_anchors=len(anchors)
        self.bbs = self.num_anchors * self.flattened_input_size
        self.bb_dimension= classes + 1 + coordinates
        self.anchor_boxes_dimension=self.num_anchors*self.bb_dimension
        #1x1 convolution to generate bbs
        self.predictor = nn.Conv2d(c, self.anchor_boxes_dimension, 1)

        # calcuate the bounding boxes of the cells
        self.bounding_boxes=BoundingBoxes((h,w),anchors)

        if use_cuda:
            self.bounding_boxes.cuda()




    # Convert a set of C features maps of HxW cells
    # to a list of HxWxA bounding boxes
    # Where A is the number of anchors per cell to use
    # Each bounding box has dimension 4+1+K
    # Dims 0:4   = x,y,h,w of the bounding box (x,y are the centers)
    # Dim 4      = objectness score
    # Dims 5:5+K = class scores, one for each class
    def feature_maps_to_bbs(self,x):
        # print("Predictor output",x.shape)
        # generate as many channels as the dimension required
        # for the bounding boxes
        x = self.predictor(x)
        batch_size = x.size(0)
        # flatten spatial dimension
        x = x.view(batch_size, self.anchor_boxes_dimension, self.flattened_input_size)
        # print("flattening spatial dims",x.shape)

        # swap spatial with channel dimension
        x = x.transpose(1, 2).contiguous()
        # print("transposing spatial and channel dims",x.shape)

        # regrouping spatial and channel dims to ungroup bbs of same
        # cell and different anchor
        x = x.view(batch_size, self.bbs, self.bb_dimension)
        # print("ungrouped bbs", x.shape)
        return x

    # Apply transformations to the bounding box parameters so that
    # they can be learned more easily
    # x and y: are sigmoided and then interpreted as displacements from
    # the center of the bounding box
    # h and w: are scaled by the anchor sizes priors
    # obj and class scores are sigmoided to get them in 0-1 range
    def transform_bbs(self,x):
        # sigmoid dx,dy
        x[:, :, 0:2] = torch.sigmoid(x[:, :, 0:2]) + self.bounding_boxes.centers
        # expand box sizes by anchor size oriors
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * self.bounding_boxes.sizes
        # sigmoid object and class scores
        x[:, :, 5:] = torch.sigmoid(x[:, :, 5:])
        return x

    def forward(self,x):
        # x.shape = [N,C,H,W]
        x=self.feature_maps_to_bbs(x)
        # x.shape = [N,BBS,4+1+K]
        # print("before transformations (first 4):\n",x[0,:4,:])
        x=self.transform_bbs(x)
        # x.shape = [N,BBS,4+1+K]
        #print("after transformations (first 4):\n", x[0, :4,:])

        return x

class DetectionLoss(nn.Module):

    def __init__(self,anchors,classes,grid_size,image_size,use_cuda=False):
        super(DetectionLoss,self).__init__()
        self.image_size=torch.FloatTensor(image_size)
        self.classes=classes
        self.anchors=anchors
        self.grid_size=grid_size
        self.use_cuda=use_cuda

    # Computes the detection loss
    # gt is is a list of N elements,
    # where each element gt_i is a matrix of dims = (BBs_i,D) where D =4+1
    # and BBs_i depends on the
    # The columns are: x,y,w,h,class
    # yhat is (N,BBs,D+K)

    def forward(self,yhat,gt):
        y=self.gt_to_output(gt)
        y=torch.autograd.Variable(torch.FloatTensor(y))
        if self.use_cuda:
            y=y.cuda()
        y[:,:,4]

        n,bbs,bb_size=yhat.shape
        assert(n==len(gt))

        for i in range(n):
            gt_bbs = gt[i][:,:4]
            gt_classes=gt[i][:,4]
            gt_bbs[:,:2] /=self.image_size
            gt_bbs[:, 2:4] /= self.image_size
            bbs=y[i,:,:4]
            ious=self.calculate_iou(bbs,gt_bbs)


    # y is a list of gt vectors
    # == (n,5)
    def gt_to_output(self,gt):
        n,d = gt.shape
        h,w = self.grid_size
        a = len(self.anchors)
        bbs=a*h*w
        y = np.zeros(n,bbs, self.classes + 5)
        for i in range(n):
            bb=gt[i,:4]
            c = gt[i,5]

            ious=self.calculate_iou(bb,)

            bb_index=0
            y[i, bb_index, 0:4] = bb
            y[i, bb_index, 5  ] = 1
            y[i, bb_index, 5+c] = 1

        return y