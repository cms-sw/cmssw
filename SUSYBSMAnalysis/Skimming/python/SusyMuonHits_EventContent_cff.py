# The following comments couldn't be translated into the new config version:

# add muon hits

# add tk dedx data

# remove  some jet collection

import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
susyMuonHitsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_standAloneMuons_*_*', 
        'keep *_globalMuons_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*', 
        'keep recoTracks_TrackRefitter_*_*', 
        'keep *_dedxHitsFromRefitter_*_*', 
        'keep *_dedxTruncated40_*_*', 
        'drop recoCaloJets_midPointCone7CaloJets_*_*', 
        'drop recoCaloJets_iterativeCone7CaloJets_*_*', 
        'drop recoGenJets_iterativeCone7GenJetsNoNuBSM_*_*', 
        'drop recoGenJets_iterativeCone7GenJetsPt10_*_*', 
        'drop recoGenJets_midPointCone5GenJetsNoNuBSM_*_*', 
        'drop recoGenJets_midPointCone7GenJetsPt10_*_*', 
        'drop recoGenJets_midPointCone7GenJetsNoNuBSM_*_*')
)
AODSIMSusyMuonHitsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyMuonHitsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyMuonHitsEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMSusyMuonHitsEventContent.outputCommands.extend(susyMuonHitsEventContent.outputCommands)
susyMuonHitsEventSelection.SelectEvents.SelectEvents.append('hscpMuon')
susyMuonHitsEventSelection.SelectEvents.SelectEvents.append('hscpMET')

