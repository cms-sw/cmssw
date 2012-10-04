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
    input = cms.untracked.int32(-1)
)

##process.MessageLogger = cms.Service("MessageLogger",
##    debugModules = cms.untracked.vstring('SiPixelDigitizer'),
##    destinations = cms.untracked.vstring('cout'),
###    destinations = cms.untracked.vstring("log","cout"),
##    cout = cms.untracked.PSet(
###       threshold = cms.untracked.string('INFO')
###       threshold = cms.untracked.string('ERROR')
##        threshold = cms.untracked.string('WARNING')
##    )
###    log = cms.untracked.PSet(
###        threshold = cms.untracked.string('DEBUG')
###    )
##)
### get the files from DBS:
###process.source = cms.Source("PoolSource",
###    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/B41EF45B-D14E-DE11-BC68-001D09F24600.root','/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/682FF80C-524F-DE11-9159-001D09F2543D.root','/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/0E9B84FD-D24E-DE11-94D9-001617C3B70E.root')
###)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [
###Muons ls -ltr /scratch/jbutt/MuPt100/* | awk '{print ",\x27 file:" $9 "\x27 "}'
# 'file:/scratch/jbutt/MuPt100/3C1F82E7-6605-E111-B612-002354EF3BE2.root'
# ,'file:/scratch/jbutt/MuPt100/82C9F14B-B402-E111-87ED-002618943821.root'
###QCD ls -ltr /scratch/jbutt/QCD80120/* | awk '{print ",\x27 file:" $9 "\x27 "}'
  ' file:/scratch/jbutt/QCD80120/16B15CEC-6605-E111-91B4-002618FDA204.root'
 ,' file:/scratch/jbutt/QCD80120/423FA9C8-E606-E111-937B-002354EF3BE1.root'
 ,' file:/scratch/jbutt/QCD80120/C2ACFE6C-2603-E111-B58F-001A9281173E.root'
 ,' file:/scratch/jbutt/QCD80120/8C340B79-8002-E111-89EE-002354EF3BE0.root'
 ,' file:/scratch/jbutt/QCD80120/7441D424-1702-E111-A600-0018F3D09696.root'
 ,' file:/scratch/jbutt/QCD80120/B08F23A4-6901-E111-AA0C-002618943898.root'
####TTbar  ls -ltr /scratch/jbutt/TTbar/* | awk '{print ",\x27 file:" $9 "\x27 "}'
# ' file:/scratch/jbutt/TTbar/1676ABF5-9004-E111-9B9B-002618FDA208.root'
#,' file:/scratch/jbutt/TTbar/5C73216C-2603-E111-AB90-002618FDA279.root'
#,' file:/scratch/jbutt/TTbar/F44EC1F2-6605-E111-85FF-003048678FDE.root'
#,' file:/scratch/jbutt/TTbar/C24DF25A-B302-E111-8AC4-001A92971BCA.root'
#,' file:/scratch/jbutt/TTbar/16DA979B-1A02-E111-9C54-0018F3D095F2.root'
#,' file:/scratch/jbutt/TTbar/34C0859D-1702-E111-835F-00304867BEDE.root'
#,' file:/scratch/jbutt/TTbar/2E90C1B8-6501-E111-8C4F-001A92971B08.root'
#,' file:/scratch/jbutt/TTbar/B4E41354-6601-E111-8D1B-00304867BED8.root'
 ] );

secFiles.extend( [
####Muons
# 'file:/scratch/jbutt/MuPt100/FE1E1645-B602-E111-9729-003048679168.root'
# ,'file:/scratch/jbutt/MuPt100/C6C414E9-6605-E111-854B-002618FDA21D.root'
# ,'file:/scratch/jbutt/MuPt100/B2649223-1702-E111-8399-001A9281174A.root'
# ,'file:/scratch/jbutt/MuPt100/3CFBFCFE-7E01-E111-B24B-0018F3D095FA.root'
###QCD
  ' file:/scratch/jbutt/QCD80120/1228DEDC-2403-E111-914A-002618943896.root'
 ,' file:/scratch/jbutt/QCD80120/182CF366-2603-E111-9F72-003048D42D92.root'
 ,' file:/scratch/jbutt/QCD80120/72EE3DEF-6605-E111-8301-002618FDA248.root'
 ,' file:/scratch/jbutt/QCD80120/DC1E3FC3-E606-E111-9C3E-00304867BEE4.root'
 ,' file:/scratch/jbutt/QCD80120/EE35547C-8002-E111-A832-001A92810AAA.root'
 ,' file:/scratch/jbutt/QCD80120/94B72EB2-1602-E111-8B8A-001A92810AE4.root'
 ,' file:/scratch/jbutt/QCD80120/9C0615B6-1602-E111-B893-001A928116E0.root'
 ,' file:/scratch/jbutt/QCD80120/E45BE72A-1902-E111-9BDE-00261894389E.root'
 ,' file:/scratch/jbutt/QCD80120/4C14BA8B-6601-E111-8D01-001A92971BB8.root'
 ,' file:/scratch/jbutt/QCD80120/625F1E38-6801-E111-A244-0030486792AC.root'
 ,' file:/scratch/jbutt/QCD80120/849BD437-6801-E111-B8F1-002618943950.root'
###TTbar
# ' file:/scratch/jbutt/TTbar/4C4CEEEC-9004-E111-81CB-0026189438E4.root'
#,' file:/scratch/jbutt/TTbar/6851646C-2603-E111-A121-001BFCDBD11E.root'
#,' file:/scratch/jbutt/TTbar/8A6EC4EE-6605-E111-BAEB-0026189438A0.root'
#,' file:/scratch/jbutt/TTbar/86C08ADC-9E02-E111-BF3E-002618943916.root'
#,' file:/scratch/jbutt/TTbar/306D3C24-1802-E111-BB8C-002618943935.root'
#,' file:/scratch/jbutt/TTbar/628DBD21-1A02-E111-A5AE-0018F3D096F8.root'
#,' file:/scratch/jbutt/TTbar/D44E792A-1802-E111-BC00-001A92971B74.root'
#,' file:/scratch/jbutt/TTbar/E05D2324-1802-E111-8B0A-00248C0BE014.root'
#,' file:/scratch/jbutt/TTbar/12938537-6401-E111-A1BB-0018F3D096EE.root'
#,' file:/scratch/jbutt/TTbar/2A2223CF-6401-E111-8DAA-001A928116FA.root'
#,' file:/scratch/jbutt/TTbar/568D9338-6201-E111-9748-001A92810AEA.root'
#,' file:/scratch/jbutt/TTbar/6C1A3035-6701-E111-A24A-002618943807.root'
#,' file:/scratch/jbutt/TTbar/6CF830DB-6E01-E111-8E98-001A92810AD2.root'
#,' file:/scratch/jbutt/TTbar/B2972B10-6401-E111-AE9C-0018F3D09634.root'
#,' file:/scratch/jbutt/TTbar/C0A60C2F-6501-E111-932F-001A92810ADE.root'
#,' file:/scratch/jbutt/TTbar/CC654D60-6401-E111-8A27-00261894385A.root'
#,' file:/scratch/jbutt/TTbar/D0FCC289-6B01-E111-8AEB-001BFCDBD15E.root'
        ] )



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
process.GlobalTag.globaltag = 'MC_50_V0::All'

process.load("CondCore.DBCommon.CondDBSetup_cfi")

# To use a test DB instead of the official pixel object DB tag: 
#process.customDead = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string('sqlite_file:/afs/cern.ch/user/v/vesna/Digitizer/dead_20100901.db'), toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityRcd'), tag = cms.string('dead_20100901'))))
#process.es_prefer_customDead = cms.ESPrefer("PoolDBESSource","customDead")


process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *','keep *_*_*_DigiTest'),
#            fileName = cms.untracked.string('rfio:/castor/cern.ch/user/v/vesna/testDigis.root')
            fileName = cms.untracked.string('file:dummy.root')
)

#process.Timing = cms.Service("Timing")

#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

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

