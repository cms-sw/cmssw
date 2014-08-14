import FWCore.ParameterSet.Config as cms

# make ntuples from RECO (ie. remove RAW)

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

process.GlobalTag.globaltag = 'GR_P_V14::All'

# turn off stuff we don't want
process.l1NtupleProducer.gmtSource = cms.InputTag("none")
process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctNonIsoEmSource = cms.InputTag("none")
process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctIsoEmSource = cms.InputTag("none")
process.l1NtupleProducer.gctEnergySumsSource = cms.InputTag("none")
process.l1NtupleProducer.gctTauJetsSource = cms.InputTag("none")
process.l1NtupleProducer.rctSourceSource = cms.InputTag("none")
process.l1NtupleProducer.dttfSourceSource = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSourceSource = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSourceSource = cms.InputTag("none")
process.l1NtupleProducer.csctfStatusSourceSource = cms.InputTag("none")

process.p.remove(process.gctDigis)
process.p.remove(process.dttfDigis)
process.p.remove(process.csctfDigis)
process.p.remove(process.l1extraParticles)
process.p.remove(process.l1ExtraTreeProducer)
process.p.remove(process.l1MuonRecoTreeProducer)


