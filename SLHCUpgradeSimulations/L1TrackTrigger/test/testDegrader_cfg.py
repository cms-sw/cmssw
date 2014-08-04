import FWCore.ParameterSet.Config as cms

process = cms.Process("VTX")

#
# This runs over a file containing L1Tracks, and determines the
# event primary vertex by running the L1TkFastVertexProducer.
#


process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 10 ) )

from SLHCUpgradeSimulations.L1TrackTrigger.ttbarFiles_cfi import *



process.source = cms.Source("PoolSource",
   fileNames = ttbarFiles
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')


process.L1TrackDegrader = cms.EDProducer("L1TrackDegrader",
        L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
	degradeZ0 = cms.bool( True ),
        degradeMomentum = cms.bool( False ),
	NsigmaPT = cms.int32( 3 )
)

process.p = cms.Path( process.L1TrackDegrader )

process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')

)


	# the degraded tracks
process.Out.outputCommands.append('keep *_L1TrackDegrader_*_*')

        # the L1Tracks, clusters and stubs
process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')


process.FEVToutput_step = cms.EndPath(process.Out)




