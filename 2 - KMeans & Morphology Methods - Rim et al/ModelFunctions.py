from SegmentationFunctions import *

def modelPipeline(img_path):
    
    ID = img_path[img_path.find('/') + 1 : img_path.find('_')]
    sliceNo = img_path[img_path.find('_') + 1 : img_path.find('.')]
    
    print('\t******* SEGMENTATION PIPELINE ********')
    print('Patient:', ID + ', Slice Number:', sliceNo)
    
    im = readImg(img_path, showOutput=0)
    
    procImg = preprocessImage(im, showOutput=0)
    print('...preprocessing')
    
    fg_mask, ch_fg_mask, fg_threshold = chullForegroundMask(procImg, showOutput=0)
    print('...computing foreground mask')
    
    lung_mask, ch_lung_mask, int_heart_mask = chullLungMask(procImg, ch_fg_mask, fg_threshold, showOutput=0)
    print('...computing lung mask')
    
    spine_mask, heart_mask = chullSpineMask(im, int_heart_mask, showOutput=0)
    print('...computing spine & heart masks')
    
    segmented_heart = segmentHeart(im, heart_mask, showOutput=0)
    
    segmented_lungs = segmentLungs(im, lung_mask, showOutput=0)
    
    heart_colored, lung_colored, colored_masks = getColoredMasks(im, heart_mask, lung_mask, showOutput=1)

slices, PatientID = readSortedSlices('sample-dataset')

for slicePath in slices:    
    modelPipeline(slicePath)
