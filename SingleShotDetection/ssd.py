import torch.nn as nn

from module import VGGBase, AuxiliaryConvolutions, PredictionConvolutions



class SSD300(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.n_class = n_class

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions()

        self.rescale_factors = nn.Parameter(torch.Float(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20.)

        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        fmap_dims = {'conv4_3': 38,
                     'conv7':19,
                     'conv8_2':10,
                     'conv9_2':5,
                     'conv10_2':3,
                     'conv11_2':1,
                     }

        obj_scales = {'conv4_3': 0.1,
                      'conv7':0.2,
                      'conv8_2':0.375,
                      'conv9_2':0.55,
                      'conv10_2':0.725,
                      'conv11_2':0.9,
                      }
        
        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, 0.333],
                         'conv8_2': [1., 2., 3., 0.5, 0.333],
                         'conv9_2': [1., 2., 3., 0.5, 0.333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5],,
                        }

        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                


    def detect_object(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        pass
