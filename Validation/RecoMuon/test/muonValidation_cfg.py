import FWCore.ParameterSet.Config as cms

processName = "MuonSuite"
process = cms.Process(processName)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( (
    '/store/relval/CMSSW_2_2_0/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_V9_v1/0000/10C12A24-74B9-DD11-85B2-001617DBCF6A.root',
    '/store/relval/CMSSW_2_2_0/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_V9_v1/0000/3A14ADED-B4B9-DD11-8F0B-001617E30D40.root' ,
    ))
secFiles.extend((
    '/store/relval/CMSSW_2_2_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/526D7CD6-68B9-DD11-886D-001617DBD224.root',
    '/store/relval/CMSSW_2_2_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/A2C70EEE-B4B9-DD11-8170-001617DBD316.root',
    '/store/relval/CMSSW_2_2_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/D437AC21-6FB9-DD11-BEA1-001617E30CC8.root' 
    ))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', "keep *_MEtoEDMConverter_*_"+processName),
    fileName = cms.untracked.string('validationEDM.root')
)
process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.debugModules = ['*']
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    TrackAssociator = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    TrackValidator = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    )
)
process.MessageLogger.cerr = cms.untracked.PSet(
    enable = cms.untracked.bool(False)
)

process.load('Configuration/StandardSequences/RawToDigi_cff')
process.raw2digi_step = cms.Path(process.RawToDigi)

process.load("Configuration/StandardSequences/SimulationRandomNumberGeneratorSeeds_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.MEtoEDMConverter_step = cms.Path(process.MEtoEDMConverter)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_V9::All"

#---- Validation stuffs ----#
## Default validation modules
process.load("Configuration.StandardSequences.Validation_cff")
process.validation_step = cms.Path(process.validation)
## Load muon validation modules
#process.recoMuonVMuAssoc.outputFileName = 'validationME.root'
process.muonValidation_step = cms.Path(cms.SequencePlaceholder("mix")+process.recoMuonValidation)

process.schedule = cms.Schedule(
    process.raw2digi_step,
#    process.validation_step,
    process.muonValidation_step,
    process.MEtoEDMConverter_step,process.outpath)

