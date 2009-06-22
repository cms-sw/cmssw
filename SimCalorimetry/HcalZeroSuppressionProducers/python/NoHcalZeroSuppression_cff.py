# Fragment to switch off HCAL zero suppression as an option
# by cmsDriver customisation

import FWCore.ParameterSet.Config as cms
def customise(process):

    process.load("SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisNoSuppression_cfi")

    return(process)
