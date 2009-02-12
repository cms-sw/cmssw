import FWCore.ParameterSet.Config as cms
#
# sequences for jet re-reco with the pat tuple
#

## do the jet reconstruction for kt6
from RecoJets.JetProducers.kt6CaloJets_cff import kt6CaloJets

## replace caloTowers by calibCaloTowers
kt6CaloJets.src = 'towerMaker'

## std sequence jet re-reco witht the pat tuple
patTupleJetReco = cms.Sequence(
                               kt6CaloJets
                              )
