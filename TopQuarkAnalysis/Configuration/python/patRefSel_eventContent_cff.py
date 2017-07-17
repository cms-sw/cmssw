import FWCore.ParameterSet.Config as cms

common_eventContent = [
  'keep edmTriggerResults_*_*_*'
, 'keep *_addPileupInfo_*_*'
, 'keep *_offlineBeamSpot_*_*'
, 'keep *_fixedGridRho*_*_*'
]

aod_eventContent = [
  'keep *_hltTriggerSummaryAOD_*_*'
, 'keep *_genParticles_*_*'
, 'keep *_genJets_*_*'
, 'keep *_offlinePrimaryVertices_*_*'
, 'keep *_secondaryVertices_*_*'
]

miniAod_eventContent = [
  'keep *_patTrigger_*_*'
, 'keep *_selectedPatTrigger_*_*'
, 'keep *_packedGenParticles_*_*'
, 'keep *_prunedGenParticles_*_*'
, 'keep *_slimmedGenJets_*_*'
, 'keep *_offlineSlimmedPrimaryVertices_*_*'
, 'keep *_slimmedSecondaryVertices_*_*'
]

refMuJets_eventContent = [
  'keep *_selectedMuons_*_*'
, 'keep *_signalMuons_*_*'
, 'keep *_selectedElectrons_*_*'
, 'keep *_selectedJets__*'
, 'keep *_selectedBTagJets__*'
, 'keep *_slimmedPhotons_*_*'
, 'keep *_slimmedTaus_*_*'
, 'keep *_slimmedMETs_*_*'
, 'keep *_goodOfflinePrimaryVertices_*_USER'
]
