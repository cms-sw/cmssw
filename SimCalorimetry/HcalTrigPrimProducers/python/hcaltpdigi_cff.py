# The following comments couldn't be translated into the new config version:

# if read_Ascii_LUTs is true then read Ascii LUTs via "inputLUTs" below

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cfi import *
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import *
HcalTPGCoderULUT = cms.ESProducer("HcalTPGCoderULUT",
    read_Ascii_LUTs = cms.bool(False),
    read_XML_LUTs = cms.bool(False),
    read_FG_LUTs = cms.bool(False),
    LUTGenerationMode = cms.bool(True),
    linearLUTs = cms.bool(False),
    tpScales = tpScales,
    MaskBit = cms.int32(0x8000),
    FG_HF_thresholds = cms.vuint32(17, 255),
    inputLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/inputLUTcoder_physics.dat'),
    FGLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/HBHE_FG_LUT.dat'),
    RCalibFile = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat')
)

HcalTrigTowerGeometryESProducer = cms.ESProducer("HcalTrigTowerGeometryESProducer")

run2_HCAL_2018.toModify(CaloTPGTranscoder, linearLUTs=cms.bool(True))
run2_HCAL_2018.toModify(HcalTPGCoderULUT, linearLUTs=cms.bool(True))
pp_on_AA_2018.toModify(CaloTPGTranscoder, FG_HF_thresholds = cms.vuint32(15, 19))
pp_on_AA_2018.toModify(HcalTPGCoderULUT, FG_HF_thresholds = cms.vuint32(15, 19))
