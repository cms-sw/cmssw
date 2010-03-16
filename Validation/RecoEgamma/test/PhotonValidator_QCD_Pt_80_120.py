
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
process.GlobalTag.globaltag = 'MC_3XY_V24::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal354_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


# official RelVal 354 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0004/C07EAA09-2D2C-DF11-87E7-002618943886.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0003/F26F69E0-882B-DF11-A76E-003048B95B30.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0003/EC346EF4-8D2B-DF11-B08F-001BFCDBD154.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0003/CA0CDD66-882B-DF11-8B8A-0026189438E9.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0003/B27CFE64-8E2B-DF11-8895-0018F3D09676.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0003/98F1A872-8A2B-DF11-8367-001A92971AEC.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0003/6699C8AD-8C2B-DF11-B0B9-001A92810ADE.root'
    ),

    secondaryFileNames = cms.untracked.vstring(


# official RelVal 354 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/F66E55F2-2C2C-DF11-BD04-002618943985.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/E66DD167-8E2B-DF11-93EF-003048678B72.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/C89E246A-8A2B-DF11-A705-001A92971BB4.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/B6FD0BC9-882B-DF11-B7B5-003048679006.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/9E1DEBC3-882B-DF11-B298-003048678F92.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/9AD9DE4F-882B-DF11-86D4-003048678B44.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/92F6B84C-8D2B-DF11-96C1-003048678B72.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/8AD1BB3F-8D2B-DF11-8182-0026189438FC.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/8000ACCF-8D2B-DF11-8380-003048678F9C.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/7A634A35-8C2B-DF11-8296-002618943964.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/762F1253-892B-DF11-AD66-00304866C398.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/72A1E73B-8A2B-DF11-AD2A-0018F3D09642.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/723482BA-882B-DF11-8856-003048678B8E.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/3E581452-882B-DF11-B537-0026189437F8.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/3C853B3F-8F2B-DF11-8EEF-001A928116C2.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/3C750BFE-8D2B-DF11-9E8D-001A92971B7C.root',
        '/store/relval/CMSSW_3_5_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0003/1C5FE04F-8E2B-DF11-BCF1-001A928116EE.root'

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


