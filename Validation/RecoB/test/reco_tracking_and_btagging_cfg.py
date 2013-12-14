# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("trackingandvalidation")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.options = cms.untracked.PSet(multiProcesses=cms.untracked.PSet(
#         maxChildProcesses=cms.untracked.int32(8),
#         maxSequentialEventsPerChild=cms.untracked.uint32(10)))


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
    skipEvents = cms.untracked.uint32(3),

)
process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")

process.plots = cms.Path(process.siPixelRecHits+process.siStripMatchedRecHits+process.pixelTracks+process.ckftracks_wodEdX+process.offlinePrimaryVertices+process.ak5JetTracksAssociatorAtVertex+process.btagging+process.inclusiveVertexing)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('trk.root'),
)
process.endpath= cms.EndPath(process.out)


process.GlobalTag.globaltag = 'START53_V27::All'

process.PoolSource.fileNames = [
'file:/data/arizzi/TomoGerrit/CMSSW_5_3_12_patch1/src/Validation/RecoB/test/qcdquick/btag004-unreconstructedFromttbar30-80angle.root'
]
