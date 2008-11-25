import FWCore.ParameterSet.Config as cms

processName = "MuonSuite"
process = cms.Process(processName)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( (
    '/store/user/aeverett/note2112recob/SingleMuPt100-step2//step2-SingleMuPt100_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT_40.root',
    ))
secFiles.extend( [
    '/store/user/aeverett/note2112/SingleMuPt100//SingleMuPt100_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT_40.root',
    ] )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', "keep *_MEtoEDMConverter_*_"+processName),
    fileName = cms.untracked.string('validationEDM.root')
)
process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories = ['TrackAssociator', 'TrackValidator']
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
    placeholder = cms.untracked.bool(True)
)


process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.MEtoEDMConverter_step = cms.Path(process.MEtoEDMConverter)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.GlobalTag.globaltag = "IDEAL_V9::All"

#---- Validation stuffs ----#
## Default validation modules
process.load("Configuration.StandardSequences.Validation_cff")
process.validation_step = cms.Path(process.validation)
## Load muon validation modules
process.load("Validation.RecoMuon.muonValidation_cff")

## Redo the DigiLinks and TrackingParticles
process.load("SimTracker.Configuration.SimTracker_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

#process.RandomNumberGeneratorService.restoreStateLabel = ""
process.RandomNumberGeneratorService.simSiPixelDigis = cms.PSet(
    initialSeed = cms.untracked.uint32(1234567),
    engineName = cms.untracked.string('HepJamesRandom'),
    )
process.RandomNumberGeneratorService.simSiStripDigis = cms.PSet(
    initialSeed = cms.untracked.uint32(1234567),
    engineName = cms.untracked.string('HepJamesRandom'),
    )


process.TPLink = cms.Sequence(process.mix*process.mergedtruth*process.trDigi) 
process.TPLink_step = cms.Path(process.TPLink)

#process.recoMuonVMuAssoc.outputFileName = 'validationME.root'

process.postMuon_step = cms.Path(process.muonSelector_seq*process.muonAssociation_seq)
process.muonValidation_step = cms.Path(process.muonValidation_seq)

process.schedule = cms.Schedule(
    process.TPLink_step,
    process.postMuon_step,
#    process.validation_step,
    process.muonValidation_step,
    process.MEtoEDMConverter_step,process.outpath)

