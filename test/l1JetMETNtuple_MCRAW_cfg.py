import FWCore.ParameterSet.Config as cms

from L1TriggerDPG.L1Ntuples.l1Ntuple_MC_cfg import *

### customisation for JetMET studies ###
### this version runs RECO and emulators on RAW MC ###


### event selection ###
# good vertices
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

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
process.load('L1TriggerDPG.L1Ntuples.L1EmulatorTree_cff')


### customise ntuple content ###
process.l1NtupleProducer.dttfSource           = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource       = cms.InputTag("none")  
process.l1NtupleProducer.csctfLCTSource       = cms.InputTag("none") 
process.l1NtupleProducer.csctfStatusSource    = cms.InputTag("none")
process.l1NtupleProducer.csctfDTStubsSource   = cms.InputTag("none")

process.l1EmulatorTree.gtSource               = cms.InputTag("none")

process.l1RecoTreeProducer.jetptThreshold = cms.double(5)
process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
process.l1extraParticles.centralBxOnly = cms.bool(False)

process.l1RecoTreeProducer.jetTag             = cms.untracked.InputTag("ak5CaloJetsL2L3")


### redefine path with the pieces we want ###
process.p = cms.Path(
    process.primaryVertexFilter
    +process.noscraping
    +process.HBHENoiseFilter
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

process.schedule = cms.Schedule(
    process.raw2digi_step,
    process.L1Reco_step,
    process.reconstruction_step,
    process.p
)


# global tag
process.GlobalTag.globaltag = 'START38_V14::All'

# N events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# input
readFiles.extend( [
    'file:/tmp/jbrooke/QCDPt0-15_Fall10_GEN-SIM-RAW.root'
#    'file:/tmp/jbrooke/MinBias-PYTHIA8.root'
#    'file:/tmp/jbrooke/Run2010B_Jet_RAW.root'
#    '/store/data/Run2010B/Jet/RAW/v1/000/149/711/226671C8-3DE7-DF11-ACEB-003048F11C58.root'
#    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FE90D67B-047A-DF11-ACE6-001A64789DC8.root',
#    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FE3E05A0-037A-DF11-967C-003048635CE2.root',
#    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FE285150-047A-DF11-9AD1-003048D47A36.root',
#    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FCEBE72D-067A-DF11-81FE-003048D45FAA.root',
#    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FCD95397-067A-DF11-8372-003048D47778.root'
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0166/8258DC0D-AF69-DF11-882A-0026189437F8.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0166/06E76E18-AF69-DF11-8DA7-003048D42DC8.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/FAE07A9A-7969-DF11-92ED-002354EF3BDB.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/FA4D4132-7769-DF11-BEE5-00261894383B.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/F69C71AC-7769-DF11-B4BC-002618943960.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/F64EB00B-7869-DF11-B7D1-002618943902.root'
] )
