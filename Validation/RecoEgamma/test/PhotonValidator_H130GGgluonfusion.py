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
process.GlobalTag.globaltag = 'START37_V5::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre1_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre1 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V5-v1/0001/98F28128-266E-DF11-8A75-003048678A88.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V5-v1/0000/FAE92E8F-EB6D-DF11-B921-003048679000.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V5-v1/0000/CC6646BD-ED6D-DF11-927E-003048679228.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V5-v1/0000/846DA811-EC6D-DF11-9A3A-003048678BE6.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V5-v1/0000/721431A7-F26D-DF11-BCD1-002618943974.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V5-v1/0000/6C3BEAAF-EC6D-DF11-8EC8-002618943821.root'
        

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre1 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0001/BE1ADF40-266E-DF11-B3A7-001A92810AD6.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/FC909697-EB6D-DF11-9F63-00261894398D.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/E4FBC384-EC6D-DF11-832B-00261894383B.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/E0AE131D-EC6D-DF11-AF12-002618943868.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/D8DFD11C-EC6D-DF11-8D8D-002618943868.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/C448F49B-EB6D-DF11-936F-002618943811.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/A4BA99BB-ED6D-DF11-8C65-0018F3D096CA.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/A4348912-EC6D-DF11-83E7-0026189437FE.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/781209AD-F26D-DF11-9104-001731EF61B4.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/7244CB97-EC6D-DF11-9CE0-002618943868.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/64A08106-ED6D-DF11-9A7E-003048678FEA.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/2A04E583-F16D-DF11-97B5-003048679070.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/1AB527A0-EC6D-DF11-BE89-0018F3D096E4.root'


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
