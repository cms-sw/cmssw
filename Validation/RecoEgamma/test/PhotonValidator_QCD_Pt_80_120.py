
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
process.GlobalTag.globaltag = 'START36_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre4_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre4 QCD_Pt_80_120

        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0002/C45511EB-1638-DF11-B9F1-0030487C90D4.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/F48B8971-6C37-DF11-AD29-0030487A3C92.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/DE7ACE6D-6B37-DF11-901F-0030487CD7EA.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/B498117C-6C37-DF11-AB0D-0030487CD7EA.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/B416C6F0-6A37-DF11-B97A-0030487C778E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/9C32A854-6C37-DF11-88AC-0030487A1FEC.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/6006A849-6C37-DF11-8A2B-0030487A1884.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0001/5045317E-6B37-DF11-A3F7-00304879BAB2.root'

        
        ),
    
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 360pre4 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/78F7F3E9-1638-DF11-82BD-0030487C7392.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/E6D0B847-6C37-DF11-BFA2-0030487CD16E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/E0FDF075-6B37-DF11-A141-0030487CD13A.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/B894F868-6B37-DF11-A41E-0030487C7828.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/B8490EEA-6A37-DF11-A030-0030487D05B0.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/94F2EB31-6B37-DF11-A6FB-0030487CD718.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/84B73C3F-6C37-DF11-9A23-0030487A17B8.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/8450706D-6C37-DF11-AC51-0030487CAF0E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/7852FD53-6C37-DF11-A699-0030487CF41E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/6C70AEB5-6A37-DF11-9FCC-0030487CD6E6.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/6C29FA6E-6B37-DF11-B57F-0030487A3C92.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/48F2468B-6C37-DF11-9F66-0030487CD6DA.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/34AA436E-6B37-DF11-A055-0030487CD716.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/32E24F58-6C37-DF11-995F-0030487C8E02.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/2463AB48-6C37-DF11-BE14-0030487C5CFA.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/1EC26027-6B37-DF11-A7AC-0030487CD16E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/12F4E57A-6C37-DF11-B477-00304879FC6C.root' 

    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)


