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

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiPixelDigitizer'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
#       threshold = cms.untracked.string('INFO')
#       threshold = cms.untracked.string('ERROR')
        threshold = cms.untracked.string('WARNING')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)
# get the files from DBS:
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/B41EF45B-D14E-DE11-BC68-001D09F24600.root','/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/682FF80C-524F-DE11-9159-001D09F2543D.root','/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0007/0E9B84FD-D24E-DE11-94D9-001617C3B70E.root')
#)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [
###Muons ls -ltr /scratch/jbutt/MuPt100/* | awk '{print ",\x27 file:" $9 "\x27 "}'
# 'file:/scratch/jbutt/MuPt100/9889CC05-8261-E011-94F7-001A92811738.root '
#,'file:/scratch/jbutt/MuPt100/00AEB646-AD60-E011-A906-00261894394D.root '
###QCD ls -ltr /scratch/jbutt/QCD80120/* | awk '{print ",\x27 file:" $9 "\x27 "}'
#  ' file:/scratch/jbutt/QCD80120/F8DCFBB7-4261-E011-B677-0026189438A9.root' 
# ,' file:/scratch/jbutt/QCD80120/0A209954-B460-E011-AD7D-002618943870.root' 
# ,' file:/scratch/jbutt/QCD80120/CC353A09-9B60-E011-B000-0026189438AA.root' 
# ,' file:/scratch/jbutt/QCD80120/7CF0CF1A-9D60-E011-9BEF-002618943884.root' 
# ,' file:/scratch/jbutt/QCD80120/70890C93-A160-E011-BFF6-003048678B5E.root' 
###TTbar  ls -ltr /scratch/jbutt/TTbar/* | awk '{print ",\x27 file:" $9 "\x27 "}'
 ' file:/scratch/jbutt/TTbar/52278ED2-4161-E011-9FDE-002618943934.root' 
,' file:/scratch/jbutt/TTbar/D863DB68-BA60-E011-9121-0018F3D09702.root' 
,' file:/scratch/jbutt/TTbar/984417B1-A960-E011-98A2-001A92811736.root' 
,' file:/scratch/jbutt/TTbar/DEA836EC-9060-E011-A388-003048678D52.root' 
#,' file:/scratch/jbutt/TTbar/8294BF95-9260-E011-B485-002354EF3BD2.root' 
#,' file:/scratch/jbutt/TTbar/169EAF9F-8C60-E011-82B6-00248C0BE005.root' 
 ] );

secFiles.extend( [
####Muons
# 'file:/scratch/jbutt/MuPt100/CA754A24-6361-E011-9CC8-00304867923E.root '
#,'file:/scratch/jbutt/MuPt100/D6CFE91F-A460-E011-A535-0018F3D096BC.root '
#,'file:/scratch/jbutt/MuPt100/666B6830-A360-E011-9A15-0018F3D0961A.root '
###QCD
#  ' file:/scratch/jbutt/QCD80120/90D1BF5E-8261-E011-90FF-001A92810AA4.root' 
# ,' file:/scratch/jbutt/QCD80120/B66E35F6-B760-E011-8374-00304867924E.root' 
# ,' file:/scratch/jbutt/QCD80120/EAD73409-9F60-E011-AF29-0026189438FD.root' 
# ,' file:/scratch/jbutt/QCD80120/D820C5B6-A260-E011-A96A-002618FDA262.root' 
# ,' file:/scratch/jbutt/QCD80120/D2BDC039-A160-E011-B201-003048678E24.root' 
# ,' file:/scratch/jbutt/QCD80120/CC774519-9D60-E011-8618-002618943970.root' 
# ,' file:/scratch/jbutt/QCD80120/AC090BAB-A660-E011-826E-001A9281172C.root' 
# ,' file:/scratch/jbutt/QCD80120/8C1F0E90-9B60-E011-B8D7-001A92971B5E.root' 
# ,' file:/scratch/jbutt/QCD80120/7E592C4A-B060-E011-8FE0-00261894391C.root' 
# ,' file:/scratch/jbutt/QCD80120/749CCA7F-9960-E011-9D50-003048679166.root' 
# ,' file:/scratch/jbutt/QCD80120/6A5B1394-A060-E011-AC40-00304867906C.root' 
# ,' file:/scratch/jbutt/QCD80120/526CEC1F-9D60-E011-8286-001A92971B74.root' 
# ,' file:/scratch/jbutt/QCD80120/4EA249A0-9C60-E011-B0E0-002618943945.root' 
# ,' file:/scratch/jbutt/QCD80120/3252A9B3-9A60-E011-A3C7-00261894386C.root' 
# ,' file:/scratch/jbutt/QCD80120/1CCA2929-9D60-E011-BD31-0026189438E4.root' 
###TTbar
 ' file:/scratch/jbutt/TTbar/FCB768B7-4261-E011-B6A9-00261894396D.root' 
,' file:/scratch/jbutt/TTbar/FED391BB-BB60-E011-B5E7-0018F3D09670.root' 
,' file:/scratch/jbutt/TTbar/8E1219D2-B360-E011-AEC0-0018F3D0962E.root' 
,' file:/scratch/jbutt/TTbar/8C5378DD-BE60-E011-BCEF-001A92810AE2.root' 
,' file:/scratch/jbutt/TTbar/FA891A2E-A060-E011-B60E-00304867906C.root' 
,' file:/scratch/jbutt/TTbar/CC2DC499-A160-E011-98B2-002618943862.root' 
,' file:/scratch/jbutt/TTbar/BADE8A9F-9A60-E011-8DC6-001BFCDBD176.root' 
,' file:/scratch/jbutt/TTbar/9885A433-9560-E011-862D-0026189438A5.root' 
,' file:/scratch/jbutt/TTbar/8A62C3C1-9B60-E011-B4D3-001A92810AEE.root' 
,' file:/scratch/jbutt/TTbar/8A00EDE3-B160-E011-A4BA-0018F3D0968C.root' 
,' file:/scratch/jbutt/TTbar/762DC67E-9D60-E011-96D3-0026189438E8.root' 
,' file:/scratch/jbutt/TTbar/6A5F5891-A060-E011-B3AC-003048679150.root' 
,' file:/scratch/jbutt/TTbar/2E43CDFD-9960-E011-A5E1-00261894392C.root' 
,' file:/scratch/jbutt/TTbar/2A3EAE9F-A760-E011-AC4A-003048678B06.root' 
,' file:/scratch/jbutt/TTbar/2493B4C4-9660-E011-A92F-00304867BFAA.root' 
,' file:/scratch/jbutt/TTbar/1A057A0C-9A60-E011-8A9B-002618943986.root' 
,' file:/scratch/jbutt/TTbar/10656835-9B60-E011-8A77-0018F3D0964A.root' 
,' file:/scratch/jbutt/TTbar/084C200A-9960-E011-83F6-00261894386E.root' 
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
process.GlobalTag.globaltag = 'MC_42_V11::All'

process.load("CondCore.DBCommon.CondDBSetup_cfi")

# To use a test DB instead of the official pixel object DB tag: 
#process.customDead = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string('sqlite_file:/afs/cern.ch/user/v/vesna/Digitizer/dead_20100901.db'), toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityRcd'), tag = cms.string('dead_20100901'))))
#process.es_prefer_customDead = cms.ESPrefer("PoolDBESSource","customDead")


process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *','keep *_*_*_DigiTest'),
#            fileName = cms.untracked.string('rfio:/castor/cern.ch/user/v/vesna/testDigis.root')
            fileName = cms.untracked.string('file:dummy.root')
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
#process.p1 = cms.Path(process.mix*process.simSiPixelDigis*process.siPixelRawData*process.siPixelDigis)

# Look at cluster charge:
#process.p1 = cms.Path(process.mix*process.digis*process.siPixelRawData*process.siPixelDigis*process.pixeltrackerlocalreco)

#This process is to run the digis validation:
#process.p1 = cms.Path(process.mix*process.simSiPixelDigis*process.pixelDigisValid)

#This process is to run the digitizer:
process.p1 = cms.Path(process.mix*process.simSiPixelDigis)

process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'

