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
    input = cms.untracked.int32(10)
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


readFiles.extend( ['/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0016/CE879860-D91E-DF11-90CB-0026189438B1.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0015/F0E24763-141E-DF11-AA59-001731AF6B85.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0015/EC2B1F47-141E-DF11-9EB6-003048678FEA.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0015/CC913CB2-131E-DF11-B181-001A92971ADC.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0015/640DF968-141E-DF11-A332-001731AF65EB.root'] );

secFiles.extend( ['/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/140E8B5B-D91E-DF11-984C-0026189438B1.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/E2E2C2AB-131E-DF11-8FFA-001A9281171C.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/D6719F39-121E-DF11-803C-00261894382D.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/BC17C767-171E-DF11-AE6E-0018F3D09614.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/B0DC2F58-141E-DF11-8B05-0018F3D09612.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/A6C64FAE-131E-DF11-9496-0030486791C6.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/9A5E8AAB-131E-DF11-9417-0018F3D096D4.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/945B8B58-141E-DF11-9EAF-0018F3D096C8.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/66FDA966-141E-DF11-B753-001731AF65EB.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/0EAC2657-141E-DF11-82F4-001A92810A98.root'] )



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
process.GlobalTag.globaltag = 'MC_3XY_V21::All'

process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *','keep *_*_*_DigiTest'),
            fileName = cms.untracked.string('file:/tmp/vesna/Digis_test.root')  
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
#process.TFileService = cms.Service("TFileService", fileName = cms.string('makeNtuple.root') )
#process.p1 = cms.Path(process.mix*process.simSiPixelDigis)

#process.p1 = cms.Path(process.mix*process.digis*process.siPixelRawData*process.siPixelDigis)

process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'

