#!/usr/bin/env python

from __future__ import print_function

import logging
import os

import pandas
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor
import numpy as np


def main():
  outPath = r''

  inputCSV = os.path.join(outPath, 'testCases.csv')
  outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
  progress_filename = os.path.join(outPath, 'pyrad_log.txt')
  params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')

  # Configure logging
  rLogger = logging.getLogger('radiomics')

  # Set logging level
  # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

  # Create handler for writing to log file
  handler = logging.FileHandler(filename=progress_filename, mode='w')
  handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
  rLogger.addHandler(handler)

  # Initialize logging for batch log messages
  logger = rLogger.getChild('batch')

  # Set verbosity level for output to stderr (default level = WARNING)
  radiomics.setVerbosity(logging.INFO)

  logger.info('pyradiomics version: %s', radiomics.__version__)
  logger.info('Loading CSV')

  # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

  try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case. This is easier for iteration over
    # the input cases
    flists = pandas.read_csv(inputCSV).T
  except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

  logger.info('Loading Done')
  logger.info('Patients: %d', len(flists.columns))

  if os.path.isfile(params):
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
  else:  # Parameter file not found, use hardcoded settings instead
    settings = {}
    # settings['binWidth'] = 25
    settings['binCount'] = 128
    settings['resampledPixelSpacing'] =  [1,1,1]
    settings['interpolator'] = sitk.sitkBSpline
    settings['enableCExtensions'] = True
    # settings['force2D'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # extractor.enableInputImages(wavelet= {'level': 2})

  logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
  logger.info('Enabled features: %s', extractor.enabledFeatures)
  logger.info('Current settings: %s', extractor.settings)

  # Instantiate a pandas data frame to hold the results of all patients
  results = pandas.DataFrame()

  for entry in flists:  # Loop over all columns (i.e. the test cases)
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                entry + 1,
                len(flists),
                flists[entry]['Image'],
                flists[entry]['Mask'])

    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)

    if str(label).isdigit():
      label = int(label)
    else:
      label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = flists[entry]  # This is a pandas Series
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
        # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
        # as the values in the rows.
        result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
        featureVector = featureVector.append(result)
      except Exception:
        logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

      # To add the calculated features for this case to our data frame, the series must have a name (which will be the
      # name of the column.
      featureVector.name = entry
      # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
      # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
      # it is 'joined' with the empty data frame.
      results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

  # logger.info('Extraction complete, writing CSV')
  # # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
  # results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
  # logger.info('CSV writing complete')
  return results


if __name__ == '__main__':
  results = main()
  normvalue = np.array([[2246.313829,119.6260191],[31531273.52,17132.98837],[3.469914433,1.174401846],[70082.70953,70.75609756],[-0.104830159,-0.81923428],[-0.098472169,-0.697554866]])
  normmaxvalue = normvalue[:,0]
  normminvalue = normvalue[:,1]
  glmcoef = [-0.397164787, -0.099752049, 0.160328501, 0.395231958, 0.076746111, 0.048079125, 0.540794987]

  featurename = ['wavelet-LLL_glcm_DifferenceVariance','wavelet-LLH_glcm_ClusterProminence','original_shape_SphericalDisproportion','wavelet-LLL_glszm_SizeZoneNonUniformity','wavelet-HLL_glcm_Imc1','wavelet-LLH_glcm_Imc1']
  feature = results.loc[featurename[0]]
  feature = np.asfarray(feature, dtype=float)
  feature = (feature - normminvalue[0]) / (normmaxvalue[0] - normminvalue[0])

  for nameid in range(1, 6):
    tempfeature = results.loc[featurename[nameid]]
    tempfeature = np.asfarray(tempfeature, dtype=float)
    tempfeature = (tempfeature-normminvalue[nameid])/(normmaxvalue[nameid]-normminvalue[nameid])
    feature = np.vstack((feature,tempfeature))

  num = np.shape(feature)[1]
  b = np.ones((1, num))
  feature = np.vstack((feature, b))
  RS = np.dot(glmcoef,feature)

  pathtest = "./results/predictRS.npy"
  outfile_x = open(pathtest, 'wb')
  np.save(outfile_x, RS)
  outfile_x.close()
