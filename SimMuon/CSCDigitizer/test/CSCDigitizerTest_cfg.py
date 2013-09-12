
# Test of CSCDigitizer
# Updated for 700pre3 - Tim Cox - 12.09.2013

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

## process.GlobalTag.globaltag = "MC_38Y_V9::All"
## Non-standard, non-MC, tag because 700pre3 is under development
process.GlobalTag.globaltag = "PRE_62_V8::All"

process.load("Validation.MuonCSCDigis.cscDigiValidation_cfi")

process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")
process.load("CalibMuon.CSCCalibration.CSCChannelMapper_cfi")
process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_7_0_0_pre3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/5E813E19-8414-E311-A5CB-0025905964B4.root'
)
)

# attempt minimal mixing module w/o pu

from SimGeneral.MixingModule.aliases_cfi import * 
from SimGeneral.MixingModule.mixObjects_cfi import * 
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *


process.mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(
      mergedtruth = cms.PSet(
            trackingParticles
      )
    ),
    LabelPlayback = cms.string(' '),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(25),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    mixObjects = cms.PSet(
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService", 
     simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('TRandom3')
    ),
     mix = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('TRandom3')
   )
)

process.DQMStore = cms.Service("DQMStore")

process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")

#process.o1 = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('cscdigis.root')
#)

process.p1 = cms.Path(process.mix*process.simMuonCSCDigis*process.cscSimDigiDump)
#process.ep = cms.EndPath(process.o1)
#

