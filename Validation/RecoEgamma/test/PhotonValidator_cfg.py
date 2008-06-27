import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")


process.DQMStore = cms.Service("DQMStore");


#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/test/testIso/CMSSW_2_1_0_pre6/src/Configuration/Examples/data/reRecoSingleGammaPt35WithIsol.root')
    #                        'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/28C3DDA9-B03E-DD11-AAA9-000423D9A212.root',
    #                        'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/34A839BD-AF3E-DD11-ADFC-000423D99E46.root',
    #                        'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/4ECD9899-B23E-DD11-90A9-001617C3B6E2.root',
    #                        'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/CC46175D-B03E-DD11-8EE1-000423D9890C.root')

)




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.photonValidation)
process.schedule = cms.Schedule(process.p1)



