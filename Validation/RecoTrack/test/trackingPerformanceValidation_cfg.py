#import FWCore.ParameterSet.Config as cms

process = cms.Process("TkVal")
process.load("FWCore.MessageService.MessageLogger_cfi")

### standard includes 
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP_V1::All'
process.GlobalTag.globaltag = 'GLOBALTAG::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(NEVENT)
)
process.source = source

### validation-specific includes
#process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
#process.load("Validation.RecoTrack.cutsTPEffic_cfi")
#process.load("Validation.RecoTrack.cutsTPFake_cfi")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

### configuration MultiTrackValidator ###
process.multiTrackValidator.out = 'val.SAMPLE.root'


process.cutsRecoTracks.algorithm = cms.string('ALGORITHM')
process.cutsRecoTracks.quality = cms.string('QUALITY')
#process.cutsRecoTracks.quality = cms.string('highPurity')

process.multiTrackValidator.associators = ['TrackAssociatorByHits']
#process.multiTrackValidator.label = ['generalTracks']

process.multiTrackValidator.label = ['TRACKS']
if (process.multiTrackValidator.label[0] == 'generalTracks'):
    process.multiTrackValidator.UseAssociators = cms.bool(False)
else:
    process.multiTrackValidator.UseAssociators = cms.bool(True)
######


#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring (
#    FILENAMES
#    )
#)
#process.extend("RelValTTbar_cff")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
	'drop *', 
        'keep HLTPerformanceInfo_*_*_*'),
    fileName = cms.untracked.string('output.SAMPLE.root')
)

#process.digi2track = cms.Sequence(process.siPixelDigis*process.SiStripRawToDigis*process.trackerlocalreco*process.ckftracks*process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*process.ckftracks*process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits)
if (process.multiTrackValidator.label[0] == 'generalTracks'):
    process.only_validation = cms.Sequence(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
else:
    process.only_validation = cms.Sequence(process.cutsRecoTracks*process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)

process.p = cms.Path(process.SEQUENCE)



