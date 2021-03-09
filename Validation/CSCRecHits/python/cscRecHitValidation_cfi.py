import FWCore.ParameterSet.Config as cms
from Validation.CSCRecHits.cscRecHitPSet import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
cscRecHitValidation = DQMEDAnalyzer(
    'CSCRecHitValidation',
    cscRecHitPSet,
    doSim = cms.bool(True),
    useGEMs = cms.bool(False),
    simHitsTag = cms.InputTag("mix","g4SimHitsMuonCSCHits")
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(cscRecHitValidation, useGEMs = True)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(cscRecHitValidation, simHitsTag = "mix:MuonSimHitsMuonCSCHits")
