import FWCore.ParameterSet.Config as cms
#
# sequences for jet re-reco with the pat tuple
#

## add geometry needed for pileup corrections for jet reco
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi       import *
from Geometry.CaloEventSetup.CaloTopology_cfi       import *

## do the jet reconstruction for kt6
from RecoJets.JetProducers.kt6CaloJets_cff import kt6CaloJets

## replace caloTowers by calibCaloTowers
kt6CaloJets.src = 'towerMaker'

## std sequence jet re-reco witht the pat tuple
patTupleJetReco = cms.Sequence(
                               kt6CaloJets
                              )
