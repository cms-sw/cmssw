# python script Pixel Digitizer testing:
# author: V. Cuplov
#
# Process available:
# 1) simulation of hits in the Tracker
# 2) digitization + validation
# 3) reconstruction or clusterization + validation 
# 4) tracks + validation
#
##############################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("DigiTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Validation.TrackerHits.trackerHitsValidation_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("SimTracker.Configuration.SimTracker_cff")
process.load("Validation.TrackerDigis.trackerDigisValidation_cff")
process.load("SimG4Core.Configuration.SimG4Core_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

## For 31X use:
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
## for older releases
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")
process.load("Validation.RecoTrack.TrackValidation_cff")
process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('PixelDigisTest'),
#    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
#)

# get the files from DBS:
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/B41EF45B-D14E-DE11-BC68-001D09F24600.root','/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/682FF80C-524F-DE11-9159-001D09F2543D.root','/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/0E9B84FD-D24E-DE11-94D9-001617C3B70E.root')
#)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( ['/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0009/B41A39C8-EF9A-DF11-B13D-0026189438E9.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0008/B49977EE-E39A-DF11-AD19-001A92971BC8.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0008/66BBD7EC-E39A-DF11-BA36-001BFCDBD1BC.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0008/0AA5E55A-E49A-DF11-A795-003048679048.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0008/00F6D8E4-E39A-DF11-9D74-003048678C9A.root'] );

secFiles.extend( ['/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0009/AA330544-079B-DF11-839F-001A9281171E.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/F6ADA2EA-E39A-DF11-A1BC-003048678E2A.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/E4E9DB59-E49A-DF11-9215-0026189438B4.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/583A850E-E59A-DF11-ACEF-00261894388B.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/4CDEE251-E39A-DF11-B23E-0026189438D3.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/3EF481EB-E39A-DF11-BE69-002618943920.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/2E101EEE-E39A-DF11-9B89-0018F3D095EA.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/24178254-E59A-DF11-A4BA-002618943923.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/0CE536EB-E39A-DF11-9546-0018F3D09688.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/0AE04552-E39A-DF11-9E1D-0026189438FE.root'] )



#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(0),
#        authenticationPath = cms.untracked.string('')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiPixelQualityRcd'),
#        tag = cms.string('SiPixelBadModule_test')
#    )),
#    connect = cms.string('sqlite_file:test.db')
#)


# Choose the global tag here:
#process.GlobalTag.globaltag = 'MC_38Y_V8::All'
process.GlobalTag.globaltag = 'MC_39Y_V1::All'

process.load("CondCore.DBCommon.CondDBSetup_cfi")

# To use a test DB instead of the official pixel object DB tag: 
#process.customDead = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string('sqlite_file:/afs/cern.ch/user/v/vesna/Digitizer/dead_20100901.db'), toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityRcd'), tag = cms.string('dead_20100901'))))
#process.es_prefer_customDead = cms.ESPrefer("PoolDBESSource","customDead")


process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *','keep *_*_*_DigiTest'),
            fileName = cms.untracked.string('rfio:/castor/cern.ch/user/v/vesna/testDigis.root')
#            fileName = cms.untracked.string('file:dummy.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

# Simulation of hits in the Tracker:
#process.simhits = cms.Sequence(process.g4SimHits*process.trackerHitsValidation)

# Digitization:
process.digis = cms.Sequence(process.simSiPixelDigis*process.pixelDigisValid)

# Reconstruction or Clusterization:
#process.rechits = cms.Sequence(process.pixeltrackerlocalreco*process.pixRecHitsValid)
process.rechits = cms.Sequence(process.pixeltrackerlocalreco*process.siStripMatchedRecHits*process.pixRecHitsValid)

#process.simSiPixelDigis.ChargeVCALSmearing=True
process.tracks = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.trackingParticles*process.trackingTruthValid*process.ckftracks*process.trackerRecHitsValidation)
process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)

# To use when you want to look at Residuals and Pulls:
#process.p1 = cms.Path(process.mix*process.digis*process.siPixelRawData*process.siPixelDigis*process.rechits*process.tracks*process.trackinghits)

#process.p1 = cms.Path(process.mix*process.digis*process.siPixelRawData*process.siPixelDigis*process.pixeltrackerlocalreco)

#This process to get events to be used to make DQM occupancy maps (cmsRun DQM_Pixel_digi.py after you ran testPixelDigitizer.py):
process.p1 = cms.Path(process.mix*process.simSiPixelDigis*process.siPixelRawData*process.siPixelDigis)

# Look at cluster charge:
#process.p1 = cms.Path(process.mix*process.digis*process.siPixelRawData*process.siPixelDigis*process.pixeltrackerlocalreco)

#This process is to run the digis validation:
#process.p1 = cms.Path(process.mix*process.simSiPixelDigis*process.pixelDigisValid)

#This process is to run the digitizer:
#process.p1 = cms.Path(process.mix*process.simSiPixelDigis)

process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'

