# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("rereco2")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.options = cms.untracked.PSet(multiProcesses=cms.untracked.PSet(
        maxChildProcesses=cms.untracked.int32(15),
        maxSequentialEventsPerChild=cms.untracked.uint32(10)))



# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#parallel processing

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
#     skipEvents = cms.untracked.uint32(281),	
    fileNames = cms.untracked.vstring(),
    secondaryFileNames = cms.untracked.vstring()
)

process.GlobalTag.globaltag = 'PRE_STA71_V4::All'

#process.reco = cms.Sequence(process.siPixelRecHits+process.siStripMatchedRecHits+process.ckftracks_wodEdX+process.offlinePrimaryVertices+process.ak5JetTracksAssociatorAtVertex*process.btagging)

process.siPixelClusters = cms.EDProducer("JetCoreClusterSplitter",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    verbose     = cms.bool(True),

    )

process.IdealsiPixelClusters = cms.EDProducer(
    "TrackClusterSplitter",
    stripClusters         = cms.InputTag("siStripClusters","","RECO"),
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    useTrajectories       = cms.bool(False),
    trajTrackAssociations = cms.InputTag('generalTracks'),
    tracks                = cms.InputTag('pixelTracks'),
    propagator            = cms.string('AnalyticalPropagator'),
    vertices              = cms.InputTag('pixelVertices'),
    simSplitPixel         = cms.bool(True), # ideal pixel splitting turned OFF
    simSplitStrip         = cms.bool(False), # ideal strip splitting turned OFF
    tmpSplitPixel         = cms.bool(False), # template pixel spliting
    tmpSplitStrip         = cms.bool(False), # template strip splitting
    useStraightTracks     = cms.bool(True),
    test     = cms.bool(True)
    )

process.compare = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    )


process.p = cms.Path(process.IdealsiPixelClusters*process.siPixelClusters*process.MeasurementTrackerEvent* process.siPixelClusterShapeCache+process.siPixelRecHits+process.siStripMatchedRecHits+process.ckftracks_wodEdX+process.offlinePrimaryVertices+process.ak4JetTracksAssociatorAtVertexPF*process.btagging*process.compare)
# process.reconstruction_fromRECO)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('trk.root'),
)
process.endpath= cms.EndPath(process.out)

process.PoolSource.secondaryFileNames =[
"file:/scratch/arizzi/jetcore/samples/qcd600800/1C3550F0-ADD0-E311-897A-02163E00EAAE.root",
"file:/scratch/arizzi/jetcore/samples/qcd600800/2689DCE2-ADD0-E311-BC2D-00304896B908.root",
"file:/scratch/arizzi/jetcore/samples/qcd600800/3EEB2741-ADD0-E311-92C6-02163E00EA17.root",
"file:/scratch/arizzi/jetcore/samples/qcd600800/DCA244FB-ADD0-E311-8E60-02163E00E84E.root"
]

process.PoolSource.fileNames = [
"file:/scratch/arizzi/jetcore/samples/qcd600800/0E2F833B-D0D0-E311-84E8-02163E00C85A.root"
]

#process.PoolSource.secondaryFileNames =[
aa =[
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/04F8F8A0-3AD1-E311-9EF4-003048679188.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/34AED71C-3BD1-E311-BEF8-0025905A60EE.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/428E5CD4-7ED1-E311-A2E0-0025905A60D0.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/588BE153-49D1-E311-B731-0025905A605E.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/72BABEA5-3AD1-E311-A53D-00261894396F.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/80BE433C-49D1-E311-9EC7-0025905A6056.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/8899FFED-52D1-E311-A087-0025905A6134.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/A8F1F8E6-7ED1-E311-BD60-002590596490.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/BAEB8FF8-7ED1-E311-AA90-0025905A60B8.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V3-v1/00000/E27E91C8-88D1-E311-B2C5-002590593878.root"
]
bb =[
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-RECO/PRE_STA71_V3-v1/00000/887DEA5B-5CD1-E311-BB97-002618943923.root",
"/store/relval/CMSSW_7_1_0_pre7/RelValTTbar/GEN-SIM-RECO/PRE_STA71_V3-v1/00000/8EEC0F1F-9FD1-E311-966F-003048FFD76E.root"
]
