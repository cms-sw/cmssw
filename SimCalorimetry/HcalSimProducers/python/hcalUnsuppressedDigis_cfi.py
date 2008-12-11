import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

# make a block so other modules, such as the data mixing module, can
# also run simulation

hcalSimBlock = cms.PSet(    
    hcalSimParameters,
    doNoise = cms.bool(True),
    doHPDNoise = cms.bool(False),
    #HPDNoiseLibrary = cms.PSet(
    #   FileName = cms.FileInPath("SimCalorimetry/HcalSimAlgos/data/hpdNoiseLibrary.root"),
    #   HPDName = cms.untracked.string("HPD")
    #),
    doTimeSlew = cms.bool(True),
    doHFWindow = cms.bool(True),
    hitsProducer = cms.string('g4SimHits')
)


simHcalUnsuppressedDigis = cms.EDProducer("HcalDigiProducer",
    hcalSimBlock
)



