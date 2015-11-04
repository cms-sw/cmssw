import FWCore.ParameterSet.Config as cms

process = cms.Process("TRA")

# This runs over a file that contains stubs and the clusters
# that make the stubs. This is the default EventContent of our
# 620_SLHC samples.
#
# It runs the tracklet-based L1Tracking. Note that the tracklet-based 
# L1Tracks are already present on our samples, but you may want to re-run
# the L1Tracking, for example to relax the extrapolation windows
# used in the algorithm, in order to recover a bit of efficiency
# over electrons that brem. Or simply to benefit from the latest
# updates of the L1Tracking code - like the possibility of 
# fitting 5-parameters tracks, not constraining them to come from
# the beamspot.


process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
         '/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/Muons/NoPU/SingleMuMinus_E2023TTI_Pt_2_10_NoPU.root'
     )
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')


# ---------------------------------------------------------------------------
# -- Run the L1Tracking :

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')

process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

process.BeamSpotFromSim =cms.EDProducer("BeamSpotFromSimProducer")

# --- In case one wants to reproduce everything (of course, the tracker
#     digis must have been kept oh the file), one just needs :
# 
# process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
# process.pL1Tracks = cms.Path( process.L1TrackTrigger )
# 


# --- But here, we run the L1Track producer starting from the existing stubs :

	# --- note that the sequence FullTrackingSequence defined in 
	#     SLHCUpgradeSimulations/L1TrackTrigger/python/L1TrackingSequence_cfi.py
	#     does both 1. and 2.  lizted below.
	# ---

process.load('Configuration.StandardSequences.L1TrackTrigger_cff')

	# ----
	#
	# 1. the following will re-create a collection of L1Tracks, with
	#    the same label as the "default" collection :
	#

# ----------------------------------------------------------------------------------
# if you want to change the extrapolation window 
# ----------------------------------------------------------------------------------

#process.TTTracksFromPixelDigis.phiWindowSF = cms.untracked.double(2.0)   #  default is 1.0


# ----------------------------------------------------------------------------------
# Uncomment line below if you want to write out ASCII file for the standalone
# tracklet simulation and emulation code
# ----------------------------------------------------------------------------------

#process.TTTracksFromPixelDigis.asciiFileName = cms.untracked.string("evlist.txt")   


# ----------------------------------------------------------------------------------
# run default version of tracking
# ----------------------------------------------------------------------------------

process.TT_step = cms.Path(process.TrackTriggerTTTracks)
process.TTAssociator_step = cms.Path(process.TrackTriggerAssociatorTracks)


# ----------------------------------------------------------------------------------
# Uncomment lines below if you want to produce the integer-version of tracklet L1 tracks
# ----------------------------------------------------------------------------------

#process.TTTracksFromPixelDigisInteger = cms.EDProducer("L1FPGATrackProducer",
#                                                       fitPatternFile  = cms.FileInPath('SLHCUpgradeSimulations/L1TrackTrigger/test/fitpattern.txt'),
#                                                       memoryModulesFile  = cms.FileInPath('SLHCUpgradeSimulations/L1TrackTrigger/test/memorymodules_full.dat'),
#                                                       processingModulesFile  = cms.FileInPath('SLHCUpgradeSimulations/L1TrackTrigger/test/processingmodules_full.dat'),
#                                                       wiresFile  = cms.FileInPath('SLHCUpgradeSimulations/L1TrackTrigger/test/wires_full.dat')
#                                                )
#process.TrackTriggerTTTracksInteger = cms.Sequence(process.BeamSpotFromSim*process.TTTracksFromPixelDigisInteger)
#process.TT_step_Integer = cms.Path(process.TrackTriggerTTTracksInteger)


# ----------------------------------------------------------------------------------
	#
	# 2. if you want to create a collection of L1Tracks with a different label, for
	#    example here, TrackTriggerTTTracksLargerPhi :
	#
	#    To use these L1Tracks later, one should use :
	#    L1TrackInputTag = cms.InputTag("TrackTriggerTTTracksLargerPhi","Level1TTTracks")

#process.TTTracksFromPixelDigisLargerPhi = process.TTTracksFromPixelDigis.clone()
#process.TTTracksFromPixelDigisLargerPhi.phiWindowSF = cms.untracked.double(2.0)   #  default is 1.0
#process.TrackTriggerTTTracksLargerPhi = cms.Sequence(process.BeamSpotFromSim*process.TTTracksFromPixelDigisLargerPhi)

#process.TTTrackAssociatorFromPixelDigisLargerPhi = process.TTTrackAssociatorFromPixelDigis.clone()
#process.TTTrackAssociatorFromPixelDigisLargerPhi.TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPixelDigisLargerPhi", "Level1TTTracks") )
#process.TrackTriggerAssociatorTracksLargerPhi = cms.Sequence( process.TTTrackAssociatorFromPixelDigisLargerPhi )

#process.TT_step = cms.Path( process.TrackTriggerTTTracksLargerPhi )
#process.TTAssociator_step = cms.Path( process.TrackTriggerAssociatorTracksLargerPhi)



# ----------------------------------------------------------------------------------
# define output module
# ----------------------------------------------------------------------------------

process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_L1Tracks.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)

process.Out.outputCommands.append('keep *_*_MergedTrackTruth_*')

process.Out.outputCommands.append( 'keep *_*_*_TRA' )
process.Out.outputCommands.append('keep *_generator_*_*')
process.Out.outputCommands.append('keep *_*gen*_*_*')
process.Out.outputCommands.append('keep *_*Gen*_*_*')
process.Out.outputCommands.append('keep *_rawDataCollector_*_*')

process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')

process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')

process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis*_Level1TTTracks_*')
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis*_Level1TTTracks_*')

process.FEVToutput_step = cms.EndPath(process.Out)




