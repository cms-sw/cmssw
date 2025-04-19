# The following comments couldn't be translated into the new config version:

# if read_Ascii_LUTs is true then read Ascii LUTs via "inputLUTs" below

import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cfi import simHcalTriggerPrimitiveDigis
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import CaloTPGTranscoder
from CalibCalorimetry.CaloTPG.tpScales_cff import tpScales
from CalibCalorimetry.HcalPlugins.Hcal_PCCUpdate_cff import PCCUpdate

HcalTPGCoderULUT = cms.ESProducer("HcalTPGCoderULUT",
    read_Ascii_LUTs = cms.bool(False),
    read_XML_LUTs = cms.bool(False),
    read_FG_LUTs = cms.bool(False),
    LUTGenerationMode = cms.bool(True),
    linearLUTs = cms.bool(False),
    contain1TSHB = cms.bool(False),
    contain1TSHE = cms.bool(False),
    containPhaseNSHE = cms.double(6.0),
    containPhaseNSHB = cms.double(6.0),
    applyFixPCC = PCCUpdate.applyFixPCC,
    overrideDBweightsAndFilterHB = cms.bool(False),
    overrideDBweightsAndFilterHE = cms.bool(False),
    tpScales = tpScales,
    MaskBit = cms.int32(0x8000),
    FG_HF_thresholds = cms.vuint32(17, 255),
    inputLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/inputLUTcoder_physics.dat'),
    FGLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/HBHE_FG_LUT.dat'),
    RCalibFile = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat')
)

HcalTrigTowerGeometryESProducer = cms.ESProducer("HcalTrigTowerGeometryESProducer")

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(HcalTPGCoderULUT, linearLUTs=True)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(HcalTPGCoderULUT, FG_HF_thresholds = [15, 19])

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(HcalTPGCoderULUT, FG_HF_thresholds = [14, 19])

from Configuration.Eras.Modifier_pp_on_PbPb_run3_2023_cff import pp_on_PbPb_run3_2023
from Configuration.Eras.Era_Run3_2023_UPC_cff import Run3_2023_UPC
(pp_on_PbPb_run3_2023 | Run3_2023_UPC).toModify(HcalTPGCoderULUT, FG_HF_thresholds = [16, 19])

from Configuration.Eras.Modifier_pp_on_PbPb_run3_2024_cff import pp_on_PbPb_run3_2024
from Configuration.Eras.Era_Run3_2024_UPC_cff import Run3_2024_UPC
(pp_on_PbPb_run3_2024 | Run3_2024_UPC).toModify(HcalTPGCoderULUT, FG_HF_thresholds = [16, 19])

#placedholder values for 2025, copied from 2024
from Configuration.Eras.Modifier_pp_on_PbPb_run3_2025_cff import pp_on_PbPb_run3_2025
from Configuration.Eras.Modifier_run3_oxygen_cff import run3_oxygen
from Configuration.Eras.Era_Run3_2025_UPC_cff import Run3_2025_UPC
(pp_on_PbPb_run3_2025 | run3_oxygen | Run3_2025_UPC).toModify(HcalTPGCoderULUT, FG_HF_thresholds = [16, 19])
