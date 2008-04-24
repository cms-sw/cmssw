# The following comments couldn't be translated into the new config version:

# if read_Ascii_LUTs is true then read Ascii LUTs via "inputLUTs" below

import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cfi import *
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import *
HcalTPGCoderULUT = cms.ESProducer("HcalTPGCoderULUT",
    read_Ascii_LUTs = cms.bool(False),
    inputLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/inputLUTcoder.dat'),
    filename = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat')
)



