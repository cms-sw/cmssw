import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.categories = ['TrackAssociator', 'TrackValidator']
#process.MessageLogger.debugModules = ['*']
#process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('DEBUG'),
#    default = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackAssociator = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackValidator = cms.untracked.PSet(
#        limit = cms.untracked.int32(-1)
#    )
#)
#process.MessageLogger.cerr = cms.untracked.PSet(
#    placeholder = cms.untracked.bool(True)
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/56B92AB9-8B7E-DD11-A821-000423D6AF24.root')
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/04983078-9082-DD11-BB8C-0019DB2F3F9B.root')
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/02F50799-FB7D-DD11-80FC-000423D98DC4.root')
)

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("Validation.RecoTrack.cuts_cff")
#NB: tracks are already filtered by the generalTracks sequence
#for additional cuts use the cutsRecoTracks filter:
#default cuts for reco tracks are dummy: please replace cuts according to your use case!
#The following cut were used to produce TDR plots
#process.cutsRecoTracks.ptMin = cms.double(0.8)
#process.cutsRecoTracks.minRapidity = cms.double(-2.5)
#process.cutsRecoTracks.maxRapidity = cms.double(2.5)
#process.cutsRecoTracks.tip = cms.double(3.5)
#process.cutsRecoTracks.lip = cms.double(30)
#process.cutsRecoTracks.minHit = cms.int32(8)
#process.cutsRecoTracks.maxChi2 = cms.double(10000) #not used in tdr
#process.cutsRecoTracks.quality = cms.vstring('tight')
#process.cutsRecoTracks.algorithm = cms.vstring('ctf')

process.load("Validation.RecoTrack.MultiTrackValidator_cff")
#process.multiTrackValidator.associators = cms.vstring('TrackAssociatorByHits','TrackAssociatorByChi2')
#process.multiTrackValidator.UseAssociators = True
#process.multiTrackValidator.label = ['cutsRecoTracks']
#process.multiTrackValidator.label_tp_effic = cms.InputTag("cutsTPEffic")
#process.multiTrackValidator.label_tp_fake  = cms.InputTag("cutsTPFake")
#process.multiTrackValidator.associatormap = cms.InputTag(assoc2GsfTracks)
#process.multiTrackValidator.out = 'file.root'

# Tracking Truth and mixing module, if needed
#process.load("SimGeneral.MixingModule.mixNoPU_cfi")
#process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.cutsTPEffic+process.cutsTPFake+process.cutsRecoTracks+process.multiTrackValidator)
#process.p = cms.Path(process.multiTrackValidator)
process.ep = cms.EndPath(process.evtInfo)


