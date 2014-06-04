import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

#
# This runs over a file that already contains the L1Tracks.
#
#
# It also runs a trivial analyzer than prints the objects
# that have been created. 


#from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p2_cfi import *
from SLHCUpgradeSimulations.L1TrackTrigger.singleMuonFiles_cfi import *

process.source = cms.Source("PoolSource",
     #fileNames = minBiasFiles_p2
     #fileNames = minBiasFiles_p1
     fileNames = singleMuonFiles
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')


process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')



# ---------------------------------------------------------------------------
#
# --- Recreate the L1Tracks to benefit from the latest updates
#

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TrackingSequence_cfi")
process.pTracking = cms.Path( process.DefaultTrackingSequence )

# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
#
# ---  Produces various L1TkMuon collections - as well as "extended" L1Muons
#

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkMuonSequence_cfi")
process.pMuons = cms.Path( process.L1TkMuons )


#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_muons.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


	# the final collection of L1TkMuons
process.Out.outputCommands.append('keep *_L1TkMuonsMerge_*_*')

	# intermediate collections :
process.Out.outputCommands.append('keep *_L1TkMuons*_*_*')
process.Out.outputCommands.append('keep *_l1extraMuExtended_*_*')
process.Out.outputCommands.append('keep *_l1TkMuonsExt*_*_*')

#process.Out.outputCommands.append('keep *L1Muon*_l1extraParticles_*_*')


	# gen-level information
#process.Out.outputCommands.append('keep *_generator_*_*')
#process.Out.outputCommands.append('keep *_*gen*_*_*')
#process.Out.outputCommands.append('keep *_*Gen*_*_*')
process.Out.outputCommands.append('keep *_genParticles_*_*')


	# the L1Tracks, clusters and stubs
process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')
#process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_*')
	# the clusters and stubs
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
#process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')
#process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')

# --- to browse the genParticles, one needs to keep the collections of associators below:
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)







