import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidationSequence_cff")
process.load("Validation.RecoEgamma.photonPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'MC_3XY_V24::All'
process.GlobalTag.globaltag = 'START36_V2::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre3_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre3 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V2-v2/0001/C8EF2567-1C31-DF11-B855-0030487A1990.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V2-v2/0000/C0D3D013-9430-DF11-8CC9-0030487CD13A.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V2-v2/0000/92D3B165-CB30-DF11-A1BD-0030487A1FEC.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V2-v2/0000/761FA87B-9930-DF11-AD56-0030487CD7C0.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V2-v2/0000/1C92506B-9830-DF11-9182-0030487C7828.root'
 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre3 RelValPhotonJets_Pt_10

        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0001/7641413D-1C31-DF11-B7A6-0030487CD812.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/FE512D42-9830-DF11-93EC-00304879EDEA.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/F62AE46E-9830-DF11-96E4-0030487CD716.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/C64198C4-9330-DF11-9855-0030487CD77E.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/A4785AED-9330-DF11-A4BF-0030487A1884.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/98789C63-9C30-DF11-9605-0030487CD7B4.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/624E9979-9930-DF11-BDD8-0030487C90EE.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/34860130-9E30-DF11-8C38-0030487CD906.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/02E95AFB-9830-DF11-8DD3-0030487A3DE0.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v2/0000/0004C47D-9430-DF11-86D5-0030487CD13A.root'
    
    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonPostprocessing.eMax  = 100
photonPostprocessing.etMax = 50




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
