import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")

process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(200)
)


from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal210_pre9_SingleGammaPt35.root'


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
 
## Official RelVal 210pre9 Single Photon pt=10
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt10-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/4862131C-6D57-DD11-847F-000423D996C8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt10-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/E421F150-6D57-DD11-ABEC-000423DD2F34.root')

## Official RelVal 210pre9 Single Photon pt=35
'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt35-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/60AF541B-6D57-DD11-9EB6-001617C3B77C.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt35-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/BE093BF5-6C57-DD11-9258-001617C3B5F4.root')

## Official RelVal 210pre9 Single Photon pt=1000
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/140FD6C7-6C57-DD11-8F00-000423D94534.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/3AC9638C-6C57-DD11-AED4-001617C3B70E.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/7C2E07C4-6C57-DD11-AB1F-000423D98B5C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/EE1DBDFC-6C57-DD11-8FF0-000423D6B358.root')


)




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.photonValidation)
process.schedule = cms.Schedule(process.p1)



