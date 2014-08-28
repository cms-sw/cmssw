import FWCore.ParameterSet.Config as cms

# make ntuples fromAOD 

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

# remove stuff that depends on RAW data
process.p.remove(process.RawToDigi)
process.p.remove(process.l1GtTriggerMenuLite)
process.p.remove(process.l1MenuTreeProducer)

process.l1NtupleProducer.gmtSource            = cms.InputTag("none")
process.l1NtupleProducer.gtEvmSource          = cms.InputTag("none")
process.l1NtupleProducer.gtSource             = cms.InputTag("none")
process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctNonIsoEmSource    = cms.InputTag("none")
process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctIsoEmSource       = cms.InputTag("none")
process.l1NtupleProducer.gctEnergySumsSource  = cms.InputTag("none")
process.l1NtupleProducer.gctTauJetsSource     = cms.InputTag("none")
process.l1NtupleProducer.rctSource            = cms.InputTag("none")
process.l1NtupleProducer.dttfSource           = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfStatusSource    = cms.InputTag("none")
process.l1NtupleProducer.csctfDTStubsSource   = cms.InputTag("none")

process.l1RecoTreeProducer.superClustersBarrelTag  = cms.untracked.InputTag("none")
process.l1RecoTreeProducer.superClustersEndcapTag  = cms.untracked.InputTag("none")
process.l1RecoTreeProducer.basicClustersBarrelTag  = cms.untracked.InputTag("none")
process.l1RecoTreeProducer.basicClustersEndcapTag  = cms.untracked.InputTag("none")

# job options

process.GlobalTag.globaltag = 'GR_R_53_V7::All'

process.options = cms.untracked.PSet(
SkipEvent = cms.untracked.vstring('ProductNotFound')
				    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

readFiles = cms.untracked.vstring(
    #'/store/data/Run2012B/JetHT/AOD/PromptReco-v1/000/197/044/C6FF5B41-00BE-E111-91BC-001D09F253D4.root'
    '/store/data/Run2012B/MinimumBias/AOD/PromptReco-v1/000/193/999/3200038A-B59D-E111-9213-003048D37580.root' 
	)
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1TreeAOD.root')
)

