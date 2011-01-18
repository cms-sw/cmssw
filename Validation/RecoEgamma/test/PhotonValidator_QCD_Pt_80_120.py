
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
process.load("Validation.RecoEgamma.conversionPostprocessing_cfi")
process.load("Validation.RecoEgamma.tkConvValidator_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START310_V1::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre7_QCD_Pt_80_120.root'
photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

tkConversionValidation.OutputFileName = 'ConversionValidationRelVal3_10_0_pre7_QCD_Pt_80_120.root'
conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0103/FCC67458-45FD-DF11-AFB0-002618943836.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0101/E87E0C52-E8FC-DF11-B93A-00261894393B.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0101/B8584466-DEFC-DF11-A1E8-001A92810AEE.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0101/A4DB5410-E7FC-DF11-96FD-00261894397A.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0101/92DC5F59-E5FC-DF11-8679-002618943866.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0101/56BE56AC-E1FC-DF11-8AFE-003048D15DCA.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START310_V1-v1/0101/56679781-EBFC-DF11-A5B4-003048678D52.root'
     ),
    
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0102/8CB4862F-39FD-DF11-AD0A-0018F3D096D8.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/FA771B8C-E9FC-DF11-B821-001A92810AE0.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/E4E564D4-DCFC-DF11-9C46-00261894391D.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/C415DA82-EFFC-DF11-8CE8-001BFCDBD1BE.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/AE316F7D-EBFC-DF11-8785-00261894396E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/A2F2220F-E7FC-DF11-925C-002618FDA28E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/A0070117-E4FC-DF11-AFB2-0018F3D0970A.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/940F4CEE-E8FC-DF11-A7D1-003048678D78.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/82341469-DEFC-DF11-B3EE-0018F3D0969C.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/7CB9A711-E7FC-DF11-9054-00261894394B.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/76EAD4AF-E1FC-DF11-8DD7-0018F3D0962C.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/705F7C90-E0FC-DF11-9E2D-0026189438CF.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/426C0021-E3FC-DF11-BEBB-0026189438ED.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/3E18E347-E6FC-DF11-B62F-001A92971B62.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/3C952051-E8FC-DF11-8033-003048678AE4.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/3630725E-E5FC-DF11-ADF6-001A92971BB8.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/302C2CE2-DEFC-DF11-9ED4-001BFCDBD19E.root'


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

process.p1 = cms.Path(process.tpSelection*process.photonPrevalidationSequence*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)


