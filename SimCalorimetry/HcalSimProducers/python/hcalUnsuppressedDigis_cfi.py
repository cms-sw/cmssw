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
    doIonFeedback = cms.bool(True),
    doThermalNoise = cms.bool(True),
    HBHEUpgradeQIE = cms.bool(False),
    HFUpgradeQIE   = cms.bool(False),
    doTimeSlew = cms.bool(True),
    doHFWindow = cms.bool(False),
    hitsProducer = cms.string('g4SimHits'),
    DelivLuminosity = cms.double(0),
    TestNumbering = cms.bool(False),
    HEDarkening = cms.bool(False),
    HFDarkening = cms.bool(False),
    minFCToDelay=cms.double(5.) # old TC model! set to 5 for the new one
)

from Configuration.StandardSequences.Eras import eras
eras.fastSim.toModify( hcalSimBlock, hitsProducer=cms.string('famosSimHits') )
