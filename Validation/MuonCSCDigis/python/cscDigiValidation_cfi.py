import FWCore.ParameterSet.Config as cms
from Validation.MuonHits.muonSimHitMatcherPSet import *
from Validation.MuonGEMDigis.muonGEMDigiPSet import *
from Validation.MuonCSCDigis.muonCSCDigiPSet import *
from Validation.MuonCSCDigis.muonCSCStubPSet import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
cscDigiValidation = DQMEDAnalyzer(
    'CSCDigiValidation',
    muonSimHitMatcherPSet,
    muonGEMDigiPSet,
    muonCSCDigiPSet,
    muonCSCStubPSet,
    simHitsTag = cms.InputTag("mix", "g4SimHitsMuonCSCHits"),
    doSim = cms.bool(True),
    useGEMs = cms.bool(False),
    ## numbering follows the chamberType in CSCDetId
    etaMins = cms.vdouble(2.0, 1.6, 1.2, 0.9, 1.6, 1.0, 1.7, 1.1, 1.8, 1.2),
    etaMaxs = cms.vdouble(2.4, 2.2, 1.7, 1.1, 2.4, 1.6, 2.4, 1.7, 2.4, 1.8)
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(cscDigiValidation, useGEMs = True)

## do not run GEMs in fastsim sequences
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(cscDigiValidation,
                 simHitsTag = "mix:MuonSimHitsMuonCSCHits",
                 simTrack = dict(inputTag = "fastSimProducer"),
                 simVertex = dict(inputTag = "fastSimProducer"),
                 cscSimHit = dict(inputTag = "MuonSimHits:MuonCSCHits")
)
