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
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START42_V6::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(100)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *

photonValidation.OutputFileName = 'PhotonValidationRelVal4_2_0_pre7_H130GGgluonfusion.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V6-v2/0037/B0E38AE3-444F-E011-8D48-0026189438B1.root',
    '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V6-v2/0032/E0BBE535-BD4E-E011-BFCA-003048678B0C.root',
    '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V6-v2/0032/CA96F351-BA4E-E011-B702-00304867BEE4.root',
    '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V6-v2/0032/4AA4406E-B44E-E011-BF87-00304867929E.root'


    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0033/6422E6B7-1D4F-E011-9913-002618FDA207.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/EEF45D5C-B44E-E011-9252-001A92811748.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/DA157FC9-B44E-E011-941A-003048678FE0.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/B8DB7DCC-B44E-E011-908B-002354EF3BDB.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/B6F92CB8-BC4E-E011-9DC5-003048678B72.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/A27AC74E-BA4E-E011-813D-001A92971BB4.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/A20609DA-BB4E-E011-803B-001A92810AE0.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/90FF6AF0-BB4E-E011-A3C2-0026189438E0.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/56C312DB-B34E-E011-956B-003048D3C010.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/4C8D35D6-B24E-E011-9151-003048679030.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/4221FF52-B54E-E011-A872-0026189438CC.root',
        '/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/2AAC7738-BC4E-E011-A35A-00304867BFAA.root'

    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 250
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 250




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonPrevalidationSequence*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
