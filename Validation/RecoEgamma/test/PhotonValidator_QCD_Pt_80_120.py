
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
process.GlobalTag.globaltag = 'START36_V4::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre6_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre6 QCD_Pt_80_120

        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0011/3E613F4E-4D45-DF11-AA1D-002618943899.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/EA44DE99-9744-DF11-BEFA-00261894394B.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/E6A65363-9B44-DF11-9A5C-0018F3D09670.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/58B8392D-9A44-DF11-BE2F-001A92971B32.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/4E901810-9744-DF11-BA42-00304867BED8.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/4871EC32-9D44-DF11-A8BD-00261894398B.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/0859AC86-9644-DF11-A03C-0026189438BC.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0010/006B6AC8-9B44-DF11-909F-003048678B36.root'

        ),
    
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 360pre6 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/781B722C-4D45-DF11-B6F8-002618943899.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/F6EF818B-9644-DF11-A9B9-003048678BE6.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/F0399D5E-9B44-DF11-BF34-0018F3D0965A.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/E63621FF-9744-DF11-A5C1-002618943921.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/DA4D69F4-9544-DF11-B487-001A92971B74.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/CE7F1B27-9A44-DF11-8D5E-001A92810AB6.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/B4629180-9844-DF11-8725-003048678B20.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/76E1020C-9744-DF11-8ABF-003048D15D04.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/704AB72A-9D44-DF11-953A-002618943947.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/6C990A71-9544-DF11-A80D-0026189438B1.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/58C5291C-9A44-DF11-BCD4-0018F3D09678.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/543CBA2B-9B44-DF11-BCE7-0018F3D095EA.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/52BCAEC2-9B44-DF11-857F-0026189438E3.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/4CF33F9B-9744-DF11-A121-001A92810AE6.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/400EA2C1-9B44-DF11-AF0F-001A92810AA2.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/2E8791AB-9C44-DF11-B76A-0018F3D0962C.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/062D4205-9744-DF11-9F08-0030486791C6.root'

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


