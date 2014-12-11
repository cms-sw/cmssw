import FWCore.ParameterSet.Config as cms

from Validation.Configuration.globalValidation_cff import *
from Validation.RecoHI.TrackValidationHeavyIons_cff import *
from Validation.RecoJets.JetValidationHeavyIons_cff import *
from Validation.RecoHI.muonValidationHeavyIons_cff import *

# change track label for rechits
hiTracks = 'hiGeneralTracks'
PixelTrackingRecHitsValid.src = hiTracks
StripTrackingRecHitsValid.trajectoryInput = hiTracks

# change ecal labels for basic clusters and super-clusters
egammaBasicClusterAnalyzer.barrelBasicClusterCollection = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters")
egammaBasicClusterAnalyzer.endcapBasicClusterCollection = cms.InputTag("islandBasicClusters","islandEndcapBasicClusters")
egammaSuperClusterAnalyzer.barrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
egammaSuperClusterAnalyzer.barrelRawSuperClusterCollection = cms.InputTag("islandSuperClusters","islandBarrelSuperClusters")
egammaSuperClusterAnalyzer.barrelCorSuperClusterCollection = cms.InputTag("correctedIslandBarrelSuperClusters")
egammaSuperClusterAnalyzer.endcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
egammaSuperClusterAnalyzer.endcapRawSuperClusterCollection = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters")
egammaSuperClusterAnalyzer.endcapCorSuperClusterCollection = cms.InputTag("correctedIslandEndcapSuperClusters")
#egammaSuperClusterAnalyzer.endcapPreSuperClusterCollection = cms.InputTag("islandEndcapSuperClustersWithPreshower") #to be implemented: only multi5x5 for now


# prevalidation sequence for all EDFilters and EDProducers
globalPrevalidationHI = cms.Sequence(
    hiTrackPrevalidation
  * hiRecoMuonPrevalidation

)


globalValidationHI = cms.Sequence(
    trackerHitsValidation      
    #+ trackerDigisValidation   # simSiDigis not in RAWDEBUG
    + trackerRecHitsValidation 
    + trackingTruthValid        
    + trackingRecHitsValid        

    + ecalSimHitsValidationSequence 
    #+ ecalDigisValidationSequence  # simEcalDigis not in RAWDEBUG
    + ecalRecHitsValidationSequence 
    + ecalClustersValidationSequence

    + hcalSimHitStudy
    #+ hcalDigisValidationSequence  # simHcalDigis not in RAWDEBUG
    + hcalRecHitsValidationSequence
    + calotowersValidationSequence
    
    + hiTrackValidation         # validation of 'hiGeneralTracks'
    + hiJetValidation           # validation of pileup jet finders
    + hiRecoMuonValidation      # validation of offline muon reco
   
    )

