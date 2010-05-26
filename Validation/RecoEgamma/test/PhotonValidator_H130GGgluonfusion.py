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
process.GlobalTag.globaltag = 'MC_3XY_V15::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre3_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


# official RelVal 350pre3 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/FC601D3E-D103-DF11-9CA5-003048D374F2.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/DE7328CB-F402-DF11-A3F4-0030487C5CFA.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/749D1328-F202-DF11-BAEA-000423D944FC.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/54F94DBC-F102-DF11-8DCE-000423D98EC8.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/407770DD-F302-DF11-AFA9-001617E30D40.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/36DAF4FB-F202-DF11-B255-0030487A322E.root'
     
    ),
    secondaryFileNames = cms.untracked.vstring(


# official RelVal 350pre3 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/EE18EFBB-F102-DF11-A880-003048D2BE08.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/E258ED9D-F102-DF11-A89E-0030487C6090.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/BC2142E0-F302-DF11-A7B0-0030487C6062.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/A25026A9-F502-DF11-8A36-00304879FA4A.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/94EE4EB6-F302-DF11-9BC6-001D09F24600.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/868F88FF-F102-DF11-A488-000423D99BF2.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/86007F4B-F302-DF11-BB69-000423D9853C.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/629F4C89-F402-DF11-A287-0030487C6090.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/523022DE-F302-DF11-8944-00304879FBB2.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/264EB6FA-F102-DF11-99F6-000423D9870C.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/18DC2499-F102-DF11-8D90-000423D98B08.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/0CD4F5D5-F202-DF11-A6D3-001617C3B76A.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/025AB388-F202-DF11-A7DD-001617E30D4A.root'

    
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
