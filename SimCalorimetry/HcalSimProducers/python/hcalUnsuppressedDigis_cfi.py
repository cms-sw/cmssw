import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

# make a block so other modules, such as the data mixing module, can
# also run simulation

hcalSimBlock = cms.PSet(    
    hcalSimParameters,
    # whether cells with MC signal get noise added
    doNoise = cms.bool(True),
    HcalPreMixStage1 = cms.bool(False),
    HcalPreMixStage2 = cms.bool(False),
    # whether cells with no MC signal get an empty signal created
    # These empty signals can get noise via the doNoise flag
    doEmpty = cms.bool(True),
    doHPDNoise = cms.bool(False),
    doIonFeedback = cms.bool(True),
    doThermalNoise = cms.bool(True),
    HBHEUpgradeQIE = cms.bool(False),
    HFUpgradeQIE   = cms.bool(False),
    #HPDNoiseLibrary = cms.PSet(
    #   FileName = cms.FileInPath("SimCalorimetry/HcalSimAlgos/data/hpdNoiseLibrary.root"),
    #   HPDName = cms.untracked.string("HPD")
    #),
    doTimeSlew = cms.bool(True),
    doHFWindow = cms.bool(False),
    hitsProducer = cms.string('g4SimHits'),
    injectTestHits = cms.bool(False),
    ChangeResponse = cms.bool(False),
    CorrFactorFile = cms.FileInPath("SimCalorimetry/HcalSimProducers/data/calor_corr01.txt"),
    DelivLuminosity = cms.double(0),
    TestNumbering = cms.bool(False),
    HEDarkening = cms.bool(False),
    HFDarkening = cms.bool(False),
    minFCToDelay=cms.double(5.) # old TC model! set to 5 for the new one
)

from Configuration.StandardSequences.Eras import eras
eras.fastSim.toModify( hcalSimBlock, hitsProducer=cms.string('famosSimHits') )
