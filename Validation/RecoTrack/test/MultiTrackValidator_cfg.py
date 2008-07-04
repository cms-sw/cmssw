import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

process.load("FWCore.MessageService.MessageLogger_cfi")
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

# Track Associators
process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

# Filters: 
process.load("Validation.RecoTrack.cuts_cff")
#NB: tracks are already filtered by the generalTracks sequence
#for additional cuts use the cutsRecoTracks filter:
#default cuts for reco tracks are dummy: please replace cuts according to your use case!
#The following cut were used to produce TDR plots
#cutsRecoTracks.ptMin = cms.double(0.8)
#cutsRecoTracks.minRapidity = cms.double(-2.5)
#cutsRecoTracks.maxRapidity = cms.double(2.5)
#cutsRecoTracks.tip = cms.double(3.5)
#cutsRecoTracks.lip = cms.double(30)
#cutsRecoTracks.minHit = cms.int32(8)
#cutsRecoTracks.maxChi2 = cms.double(10000) #not used in tdr
#cutsRecoTracks.quality = cms.string('tight')
#cutsRecoTracks.algorithm = cms.string('ctf')

#cutsTPEffic.src = cms.InputTag("trackingtruthprod")
#cutsTPFake.src = cms.InputTag("trackingtruthprod")

# Track Validator    
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
#multiTrackValidator.associators = cms.vstring('TrackAssociatorByHits','TrackAssociatorByChi2')
#multiTrackValidator.UseAssociators = cms.bool(True)
#multiTrackValidator.label = cms.VInputTag(cms.InputTag(cutsRecoTracks))
#multiTrackValidator.associatormap = cms.InputTag(assoc2GsfTracks) 
#multiTrackValidator.out = cms.string('file.root')
    
# Tracking Truth and mixing module, if needed
#process.load("SimGeneral.MixingModule.mixNoPU_cfi")
#process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/cerati/TTbar.root')
)

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.multiTrackValidator)
process.ep = cms.EndPath(process.evtInfo)


