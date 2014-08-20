import FWCore.ParameterSet.Config as cms

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

### customisation for JetMET studies ###
### this version runs RECO and emulators on RAW data ###


### event selection ###
# good vertices
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)
process.load('L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi')
# track quality filter
process.noscraping = cms.EDFilter("FilterOutScraping",
  applyfilter = cms.untracked.bool(True),
  debugOn = cms.untracked.bool(False),
  numtrack = cms.untracked.uint32(10),
  thresh = cms.untracked.double(0.25)
)

# HCAL noise filter
process.load('CommonTools/RecoAlgos/HBHENoiseFilter_cfi')
process.hbhefilter = cms.Path(process.HBHENoiseFilter)


### emulator ###

process.load('L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi')
process.load('L1TriggerDPG.L1Ntuples.L1EmulatorTree_cff')

### HLT Filter ###

### customise ntuple content ###
process.l1NtupleProducer.generatorSource      = cms.InputTag("none")
process.l1NtupleProducer.simulationSource     = cms.InputTag("none")
process.l1NtupleProducer.dttfSource           = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfStatusSource    = cms.InputTag("none")
process.l1EmulatorTree.gtSource               = cms.InputTag("none")

process.l1RecoTreeProducer.jetptThreshold = cms.double(5)
process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
process.l1extraParticles.centralBxOnly = cms.bool(False)


process.load("L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi")
#Dump the config info
process.load("L1TriggerConfig.GctConfigProducers.l1GctConfigDump_cfi")



### redefine path with the pieces we want ###
process.p = cms.Path(
    #process.HLTFilter
    #process.primaryVertexFilter
    process.noscraping
    #+process.HBHENoiseFilter
    +process.l1NtupleProducer
    +process.l1ExtraTreeProducer
    +process.l1RecoTreeProducer
    +process.valRctDigis
    +process.valGctDigis
    +process.valGmtDigis
    +process.valL1extraParticles
    +process.l1EmulatorTree
    +process.l1EmulatorExtraTree
)

### reconstruction
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)

# jet corrections

process.schedule = cms.Schedule(
    process.raw2digi_step,
    process.L1Reco_step,
    process.reconstruction_step,
    process.p
)


# global tag
#process.GlobalTag.globaltag = 'GR_R_42_V19::All'

# N events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# input
readFiles.extend( [
"/store/data/Run2011A/Jet/RAW/v1/000/168/162/D65F541D-19A3-E011-A074-485B39897227.root",
"/store/data/Run2011A/Jet/RAW/v1/000/168/229/8E741C7C-34A3-E011-8C3A-003048F117EA.root"
# '/store/data/Run2011A//MinimumBias/RAW/v1/000/161/217/E4F09BE9-E654-E011-8F8A-003048F118D2.root',
])
