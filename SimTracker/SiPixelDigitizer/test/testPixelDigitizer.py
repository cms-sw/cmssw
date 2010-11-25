2# python script Pixel Digitizer testing:
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


readFiles.extend( ['/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0090/AE814955-91F5-DF11-8C95-003048678B20.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0087/8C30A969-72F5-DF11-A537-0018F3D096C8.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0085/C2E568D8-0DF5-DF11-9EA2-0026189437F5.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0085/9A402DD9-15F5-DF11-87D6-003048D15E24.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0085/28199BD0-12F5-DF11-9A35-001A92971B5E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0085/023672FD-0DF5-DF11-AE40-003048678CA2.root'] );

secFiles.extend( ['/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0090/06D49A63-91F5-DF11-B2CE-00261894380B.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0086/E0F513AA-2BF5-DF11-863B-001A92810A96.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/F85C3C87-0BF5-DF11-ABEC-001A92971B8C.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/CC8F5B4B-17F5-DF11-8419-002354EF3BDF.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/C046BD12-11F5-DF11-BBE4-0026189438A7.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/5A335988-14F5-DF11-805C-003048679048.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/4C5E3F2D-12F5-DF11-B24C-0018F3D096A6.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/4A2036B3-07F5-DF11-A8AC-001A92971BDC.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/385FE049-0DF5-DF11-BFE1-00248C0BE016.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/08E5C3EC-12F5-DF11-9B5C-003048679084.root'] )



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
process.GlobalTag.globaltag = 'MC_39Y_V5::All'

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

