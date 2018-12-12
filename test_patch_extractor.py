#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""

import patch_extractor as patchX

#use your own import function for data
mri = patchX.dataset_loader("dataset/mri/",["P004","P020"],"_b1000.nrrd")
roi = patchX.dataset_loader("dataset/roi/",["P004","P020"],"_ROI.nrrd")

patches_with_roi, patches_no_roi = patchX.get_patches(mri,roi,patch_size=16,thresshold=0.2)
