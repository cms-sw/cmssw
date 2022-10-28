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
    # numbering follows the chamberType in CSCDetId
    etaMins = cms.vdouble(2.0, 1.6, 1.2, 0.9, 1.6, 1.0, 1.7, 1.1, 1.8, 1.2),
    etaMaxs = cms.vdouble(2.4, 2.2, 1.7, 1.1, 2.4, 1.6, 2.4, 1.7, 2.4, 1.8),
    # variables for extra diagnostic plots
    # largely copied from DQM/L1TMonitor/python/L1TdeCSCTPG_cfi.py
    chambers = cms.vstring("ME11", "ME12", "ME13", "ME21", "ME22",
                           "ME31", "ME32", "ME41", "ME42"),
    # which chambers are running the Run-3 algorithm already?
    # currently ME1/1, ME2/1, ME3/1 and ME4/1
    # ME1/3 and MEX/2 not configured with Run-3 algorithm from start of data taking
    chambersRun3 = cms.vuint32(0, 3, 5, 7),
    alctVars = cms.vstring("quality", "wiregroup", "bx"),
    alctNBin = cms.vuint32(6, 116, 16),
    alctMinBin = cms.vdouble(0, 0, 0),
    alctMaxBin = cms.vdouble(6, 116, 16),
    clctVars = cms.vstring(
        # For Run-2 eras
        "quality", "halfstrip", "pattern", "bend", "bx",
        # Added in Run-3 eras
        # The quartstripbit and eighthstripbit are left out to prevent
        # too many plots being created by the release validation
        "quartstrip", "eighthstrip", "run3pattern",
        "slope", "compcode"),
    clctNBin = cms.vuint32(6, 224, 16, 2, 16, 448, 896, 5, 16, 410),
    clctMinBin = cms.vdouble(3, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    clctMaxBin = cms.vdouble(9, 224, 16, 2, 16, 448, 896, 5, 16, 410),
    preclctVars = cms.vstring(
        # for pretriggers we look at a reduced set of variables
        "cfeb", "halfstrip", "bx"),
    preclctNBin = cms.vuint32(7, 224, 16),
    preclctMinBin = cms.vdouble(0, 0, 0),
    preclctMaxBin = cms.vdouble(7, 224, 16),
    lctVars = cms.vstring(
        # For Run-2 eras
        "quality", "wiregroup", "halfstrip", "pattern", "bend", "bx", "type",
        # Added in Run-3 eras
        # The quartstripbit and eighthstripbit are left out to prevent
        # too many plots being created by the release validation
        "quartstrip", "eighthstrip", "run3pattern", "slope"),
    lctNBin = cms.vuint32(16, 116, 224, 16, 2, 16, 8, 448, 896, 5, 16),
    lctMinBin = cms.vdouble(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    lctMaxBin = cms.vdouble(16, 116, 224, 16, 2, 16, 8, 448, 896, 5, 16),
    isRun3 = cms.bool(False),
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(cscDigiValidation,
                     isRun3 = True
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(cscDigiValidation,
                  useGEMs = True
)

# do not run GEMs in fastsim sequences
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(cscDigiValidation,
                 simHitsTag = "mix:MuonSimHitsMuonCSCHits",
                 simTrack = dict(inputTag = "fastSimProducer"),
                 simVertex = dict(inputTag = "fastSimProducer"),
                 cscSimHit = dict(inputTag = "MuonSimHits:MuonCSCHits")
)
