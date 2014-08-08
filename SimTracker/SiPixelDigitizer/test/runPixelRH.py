# produce pixel cluster & rechits from digia
# works directly or through raw
# 
#
##############################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("ClusTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Services_cff")

# clusterizer 
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

# for raw
#process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
#process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')


# needed for pixel RecHits (templates?)
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiPixelClusterizer'),
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
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
#    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100/digis/digis1.root'
#    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/digis/digis2_postls171.root'
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/digis/digis2_mc71.root'
  )
)

# Choose the global tag here:
#process.GlobalTag.globaltag = "MC_70_V1::All"
#process.GlobalTag.globaltag = "START70_V1::All"
#process.GlobalTag.globaltag = "START71_V1::All"
process.GlobalTag.globaltag = "MC_71_V1::All"
#process.GlobalTag.globaltag = "POSTLS171_V1::All"
#process.GlobalTag.globaltag = "PRE_MC_71_V2::All"


# DB stuff 
useLocalDB = True
if useLocalDB :
# Frontier LA 
    process.DBReaderFrontier = cms.ESSource("PoolDBESSource",
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(0),
         authenticationPath = cms.untracked.string('')
     ),
     toGet = cms.VPSet(
 	 cms.PSet(
# GenError
          record = cms.string('SiPixelGenErrorDBObjectRcd'),
#          tag = cms.string('SiPixelGenErrorDBObject38Tv1')
#          tag = cms.string('SiPixelGenErrorDBObject38TV10')
#          tag = cms.string('SiPixelGenErrorDBObject38T_v0_mc1')
          tag = cms.string('SiPixelGenErrorDBObject_38T_v1_mc')
# LA
# 			record = cms.string("SiPixelLorentzAngleRcd"),
# 			label = cms.untracked.string("fromAlignment"),
# 			label = cms.untracked.string("forWidth"),
# 			tag = cms.string("SiPixelLorentzAngle_v02_mc")
# 			tag = cms.string("SiPixelLorentzAngle_fromAlignment_v0_mc")
# 			tag = cms.string("SiPixelLorentzAngle_forWidth_v0_mc")
 		),
# 		cms.PSet(
# 			record = cms.string("SiPixelLorentzAngleSimRcd"),
# 			tag = cms.string("test_LorentzAngle_Sim")
# 		)
 	),
#     connect = cms.string('frontier://FrontierProd/CMS_COND_31X_PIXEL')
     connect = cms.string('frontier://FrontierPrep/CMS_COND_PIXEL')
    ) # end process

# SQ_LITE GenError
    process.DBReaderFrontier2 = cms.ESSource("PoolDBESSource",
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(0),
         authenticationPath = cms.untracked.string('')
     ),
     toGet = cms.VPSet(
 		cms.PSet(
 			record = cms.string("SiPixelGenErrorDBObjectRcd"),
# 			label = cms.untracked.string("fromAlignment"),
# 			tag = cms.string("SiPixelGenErrorDBObject38Tv1")
 			tag = cms.string("SiPixelGenErrorDBObject38TV10")
 		),
 	),
#     connect = cms.string('sqlite_file:siPixelGenErrors38T.db_old')
     connect = cms.string('sqlite_file:siPixelGenErrors38T.db')
   ) # end process
# endif
 
process.myprefer = cms.ESPrefer("PoolDBESSource","DBReaderFrontier")
#process.myprefer2 = cms.ESPrefer("PoolDBESSource","DBReaderFrontier2")


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
#
# To use a test DB instead of the official pixel object DB tag: 
#process.customDead = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string('sqlite_file:/afs/cern.ch/user/v/vesna/Digitizer/dead_20100901.db'), toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityRcd'), tag = cms.string('dead_20100901'))))
#process.es_prefer_customDead = cms.ESPrefer("PoolDBESSource","customDead")


process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *','keep *_*_*_ClusTest'),
#            fileName = cms.untracked.string('file:clus.root')
            fileName = cms.untracked.string('file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/rechits/rechits2_mc71.root')
)

#process.Timing = cms.Service("Timing")
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

# My 
# modify clusterie parameters
#process.siPixelClusters.ClusterThreshold = 4000.0

# DIRECT
# direct clusterization (no raw step)
# label of digis 
process.siPixelClusters.src = 'simSiPixelDigis'

# plus pixel clusters  (OK)
#process.p1 = cms.Path(process.siPixelClusters)
# plus pixel rechits (OK)
process.p1 = cms.Path(process.pixeltrackerlocalreco)

# RAW
# clusterize through raw (OK)
# for Raw2digi for simulations 
#process.siPixelRawData.InputLabel = 'mix'
#process.siPixelDigis.InputLabel = 'siPixelRawData'
# process.siStripDigis.ProductLabel = 'SiStripDigiToRaw'
#process.siPixelClusters.src = 'siPixelDigis'

#process.p1 = cms.Path(process.siPixelRawData)
#process.p1 = cms.Path(process.siPixelRawData*process.siPixelDigis)
#process.p1 = cms.Path(process.siPixelRawData*process.siPixelDigis*process.pixeltrackerlocalreco)

process.outpath = cms.EndPath(process.o1)
