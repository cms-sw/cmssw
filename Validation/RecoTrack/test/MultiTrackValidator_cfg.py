import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
)


#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing"
    ,summaryOnly = cms.untracked.bool(True)
)

# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V15-v1/00000/12CAFE96-F9FE-E311-B68B-0025905964CC.root'
                  ] )


secFiles.extend( [
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/9E420320-EFFE-E311-AF6C-0026189438CC.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/1C86C22E-F1FE-E311-BFF9-002354EF3BDE.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/34143955-EFFE-E311-82E6-0025905A6104.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/44F3997B-F1FE-E311-8A94-002618FDA208.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/54B6EB62-F4FE-E311-A5CF-0025905A6104.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/60C85922-EFFE-E311-9937-002354EF3BDD.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/7A883CA4-F1FE-E311-BA87-0025905A60A0.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/8A16120E-F1FE-E311-A2E8-0025905A6110.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/A4C4137F-F2FE-E311-82BC-0025905A6136.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/B05AA00F-F2FE-E311-BA9F-002618943978.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/C0347495-EFFE-E311-9B7D-003048FFD728.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/C82CD020-EFFE-E311-9894-00261894389F.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/CA3E4B23-EFFE-E311-87D2-002618943943.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/D2F1A740-F1FE-E311-8BBD-0025905822B6.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/E4D57925-EFFE-E311-BA2E-0025905A48B2.root',
       '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_POSTLS171_V15-v1/00000/EAE281B4-EFFE-E311-96E1-0025905A6090.root'
                 ] )
process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(400) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')


### validation-specific includes
#process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')




########### configuration MultiTrackValidator ########
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
process.multiTrackValidator.skipHistoFit=cms.untracked.bool(False)
#process.cutsRecoTracks.quality = cms.vstring('','highPurity')
#process.cutsRecoTracks.quality = cms.vstring('')
process.multiTrackValidator.label = ['cutsRecoTracks']
process.multiTrackValidator.useLogPt=cms.untracked.bool(True)
process.multiTrackValidator.minpT = cms.double(0.1)
process.multiTrackValidator.maxpT = cms.double(3000.0)
process.multiTrackValidator.nintpT = cms.int32(40)
process.multiTrackValidator.UseAssociators = cms.bool(True)


#process.load("Validation.RecoTrack.cuts_cff")
#process.cutsRecoTracks.quality = cms.vstring('highPurity')
#process.cutsRecoTracks.ptMin    = cms.double(0.5)
#process.cutsRecoTracks.minHit   = cms.int32(10)
#process.cutsRecoTracks.minRapidity  = cms.int32(-1.0)
#process.cutsRecoTracks.maxRapidity  = cms.int32(1.0)

process.quickTrackAssociatorByHits.useClusterTPAssociation = cms.bool(True)
process.load("SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi")

process.validation = cms.Sequence(
    process.tpClusterProducer *
    process.multiTrackValidator
)

# paths
process.val = cms.Path(
      process.cutsRecoTracks
    * process.validation
)

# Output definition
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('file:MTV_inDQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


process.schedule = cms.Schedule(
      process.val,process.endjob_step,process.DQMoutput_step
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    wantSummary = cms.untracked.bool(True)
)



