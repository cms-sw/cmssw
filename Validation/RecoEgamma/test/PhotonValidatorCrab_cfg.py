import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidationSequence_cff")
process.load("DQMOffline.EGamma.egammaDQMOffline_cff");
process.GlobalTag.globaltag = 'MC_31X_V3::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)








from Validation.RecoEgamma.photonValidationSequence_cff import *
photonValidation.OutputMEsInRootFile = True


from DQMOffline.EGamma.photonAnalyzer_cfi import *
from DQMOffline.EGamma.piZeroAnalyzer_cfi import *

photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = photonValidation.OutputFileName
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.useTriggerFiltering = cms.bool(True)

piZeroAnalysis.standAlone = cms.bool(True)
piZeroAnalysis.OutputMEsInRootFile = cms.bool(True)
piZeroAnalysis.OutputFileName = photonValidation.OutputFileName
piZeroAnalysis.useTriggerFiltering = cms.bool(False)




process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 311 single Photons pt=10GeV
'/store/relval/CMSSW_3_1_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V2-v1/0002/A0BFDEDF-776B-DE11-9054-000423D98C20.root'
 
    )
                            
 )



photonValidation.minPhoEtCut = 10
photonValidation.eMax  = 1500
photonValidation.etMax = 1500
photonValidation.signal = True

process.photons.ecalRecHitSumEtOffsetBarrel = 99999.
process.photons.ecalRecHitSumEtSlopeBarrel = 0.
process.photons.ecalRecHitSumEtOffsetEndcap = 99999.
process.photons.ecalRecHitSumEtSlopeEndcap = 0.
process.photons.hcalTowerSumEtOffsetBarrel = 99999.
process.photons.hcalTowerSumEtSlopeBarrel = 0.
process.photons.hcalTowerSumEtOffsetEndcap = 99999.
process.photons.hcalTowerSumEtSlopeEndcap = 0.


#process.p1 = cms.Path(process.photonValidation)
process.p1 = cms.Path(process.tpSelection*process.photons*process.egammaDQMOffline*process.photonValidationSequence*process.dqmStoreStats)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



