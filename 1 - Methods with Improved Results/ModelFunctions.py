from SegmentationFunctions import *

def modelPipeline(img_path):
    
    ID = img_path[img_path.find('/') + 1 : img_path.find('_')]
    sliceNo = img_path[img_path.find('_') + 1 : img_path.find('.')]
    
    print('\t******* SEGMENTATION PIPELINE ********')
    print('Patient:', ID + ', Slice Number:', sliceNo)
    
    im = readImg(img_path, showOutput=0)
    
    procImg, fg_threshold = preprocessImage(im, showOutput=0)
    print('...preprocessing')
    
    fg_mask = getForegroundMask(procImg, fg_threshold, showOutput=0)
    print('...computing foreground mask')
    
    trachea_mask, lung_mask, ch_lung_mask, int_heart_mask = getLungTracheaMasks(procImg, 
                                                                          fg_mask, 
                                                                          fg_threshold, 
                                                                          showOutput=0)
    print('...computing lung mask')
    
    spine_mask, heart_mask = chullSpineMask(im, int_heart_mask, showOutput=0)
    print('...computing spine & heart masks')
    
    segmented_heart, segmented_lungs, segmented_trachea = segmentHeartLungsTrachea(im, 
                                                                               heart_mask, 
                                                                               lung_mask, 
                                                                               trachea_mask, 
                                                                               showOutput=0)
    
    heart_colored, lung_colored, trachea_colored, colored_masks = getColoredMasks(im, 
                                                                                  heart_mask, 
                                                                                  lung_mask, 
                                                                                  trachea_mask,
                                                                                  showOutput=1)

slices, PatientID = readSortedSlices('sample-dataset')

for slicePath in slices:    
    modelPipeline(slicePath)
