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
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")

process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(200)
)


from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal210_pre6_3.8T.root'


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'file:/data/test/testIso/CMSSW_2_1_0_pre6/src/Configuration/Examples/data/reRecoSingleGammaPt35WithIsol.root')



 ## Official RelVal 210pre6 Single Photon pt=35 GeV 4T   
  #                         'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/28C3DDA9-B03E-DD11-AAA9-000423D9A212.root',
  #                          'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/34A839BD-AF3E-DD11-ADFC-000423D99E46.root',
  #                          'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/4ECD9899-B23E-DD11-90A9-001617C3B6E2.root',
  #                          'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/CC46175D-B03E-DD11-8EE1-000423D9890C.root')

## Official RelVal 210pre6 Single Photon pt=35 GeV 3.8 T
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0004/6AF10932-F340-DD11-9BB7-000423D94700.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0004/AAFFE6D9-0141-DD11-9019-001617DBCF6A.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/245E3E90-0741-DD11-BB73-000423D99660.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/2A7A33B3-1841-DD11-8002-000423D6B42C.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/4CD97038-0741-DD11-8FD7-001D09F23174.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/9058D215-0541-DD11-BA20-000423D986C4.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/E2B6920E-0541-DD11-9144-000423D99A8E.root' )


## Official RelVal 210pre6 Single Photon pt=1000 GeV 4T   
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/10FE12B7-AF3E-DD11-9B4C-000423D6CA6E.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/4A9C5FF7-AE3E-DD11-9E71-000423D992A4.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/5294E693-AE3E-DD11-9E19-001617C3B710.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/9EFA96BC-AF3E-DD11-88B1-001617E30D06.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/A6FE0D26-AF3E-DD11-A4DE-001617C3B710.root')



)




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.photonValidation)
process.schedule = cms.Schedule(process.p1)



