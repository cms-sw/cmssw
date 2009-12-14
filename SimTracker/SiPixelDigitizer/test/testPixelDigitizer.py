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


readFiles.extend( ['/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V9A-v1/0008/E8103237-33E4-DE11-A447-001A92810AEE.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V9A-v1/0008/A4865AED-2CE4-DE11-9081-002354EF3BE2.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V9A-v1/0008/9411F4EC-2CE4-DE11-A782-003048678FE4.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V9A-v1/0008/1219A3F4-2CE4-DE11-9FCB-001A9281173A.root'] );

secFiles.extend( [ '/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/FEF402CE-9EE4-DE11-8EE5-00261894388A.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/F2F7DFEE-2CE4-DE11-99A7-001BFCDBD182.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/CADFF1EA-2CE4-DE11-B4E7-0026189438FD.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/C06C50F0-2CE4-DE11-BE00-0018F3D096F8.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/A60C53EA-2CE4-DE11-94BC-00261894394B.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/9C468D6B-2DE4-DE11-90B1-0018F3D096F8.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/920E11E4-2CE4-DE11-A58B-002354EF3BDE.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/369953EB-2CE4-DE11-8184-0026189438C2.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/32C3A59C-2BE4-DE11-81CD-0026189438E0.root','/store/relval/CMSSW_3_3_6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0008/1460206A-2DE4-DE11-A417-001A928116AE.root'] )



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
process.GlobalTag.globaltag = 'MC_3XY_V9A::All'

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

