import FWCore.ParameterSet.Config as cms

process = cms.Process("SEEDVALIDATOR")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-RECO/v1/0000/6C962D0D-7DF3-E111-A62D-001A92971ADC.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-RECO/v1/0000/405707C1-6EF3-E111-B25E-00261894395B.root'
 ] );


secFiles.extend( [
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/v1/0000/E88AC5BF-49F3-E111-B2C8-00261894395B.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/v1/0000/88A046AF-46F3-E111-A727-002618943867.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/v1/0000/7C7EB0A0-46F3-E111-9F8B-001A92971B8A.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/v1/0000/62D6C88B-46F3-E111-B851-003048678AC0.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/v1/0000/2A59578A-46F3-E111-B53E-003048FFD756.root'
    ] );

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START60_V4::All'

process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
)

#process.MessageLogger.categories = ['TrackValidator','SeedValidator','TrackAssociator']
#process.MessageLogger.debugModules = ['*']#compile with: scram b USER_CXXFLAGS="-g\ -D=EDM_ML_DEBUG"
#process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('DEBUG'),
#    default = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackValidator = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    SeedValidator = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackAssociator = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    )
#)
#process.MessageLogger.cerr = cms.untracked.PSet(
#    placeholder = cms.untracked.bool(True)
#)

process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.trackAssociatorByHits.SimToRecoDenominator = cms.string('reco')
#process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")

process.load("Validation.RecoTrack.cuts_cff")

process.load("Validation.RecoTrack.TrackerSeedValidator_cff")
process.trackerSeedValidator.associators = ['trackAssociatorByHits']
process.trackerSeedValidator.label = ['initialStepSeeds']
process.trackerSeedValidator.outputFile = 'file.root'

# Tracking Truth and mixing module, if needed
#process.load("SimGeneral.MixingModule.mixNoPU_cfi")
#process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.trackingGlobalReco
                     #*process.cutsTPEffic*process.cutsTPFake
                     *process.trackAssociatorByHits
                     *process.trackerSeedValidator)

process.ep = cms.EndPath(process.evtInfo)


