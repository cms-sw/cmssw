import FWCore.ParameterSet.Config as cms

process = cms.Process("DEG")

#
# Produces collections of L1Tracks with degraded resolutions
#   - in z0  (to reproduce the z resolutions expected with the tilted tracer)
#   - in PT  (to look at the impact of a worse PT resolution)
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


# ---------------------------------------------------------------------------
#
# --- First, recreate the L1Tracks to benefit from the latest updates

process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TrackingSequence_cfi")
process.pTracking = cms.Path( process.DefaultTrackingSequence )


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#
# --- Now, produce two collections of degraded tracks 


	# worse z0 as for the tilted tracker:
process.L1TrackDegraderZ0 = cms.EDProducer("L1TrackDegrader",
        L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks","DEG"),
	degradeZ0 = cms.bool( True ),
        degradeMomentum = cms.bool( False ),
	NsigmaPT = cms.int32( 3 )    # dummy here
)
process.pZ0 = cms.Path( process.L1TrackDegraderZ0 )


	# degrade the PT resolustion by approx. a factor of 3
process.L1TrackDegraderPT3 = cms.EDProducer("L1TrackDegrader",
        L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks","DEG"),
        degradeZ0 = cms.bool( False ),
        degradeMomentum = cms.bool( True ),
        NsigmaPT = cms.int32( 3 )   
)
process.pPT3 = cms.Path( process.L1TrackDegraderPT3 )


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_degrader.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')

)


	# the degraded tracks
process.Out.outputCommands.append('keep *_L1TrackDegrader*_*_*')

        # the L1Tracks, clusters and stubs
process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')


process.FEVToutput_step = cms.EndPath(process.Out)




