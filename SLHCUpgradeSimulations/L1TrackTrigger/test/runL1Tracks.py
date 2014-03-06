import FWCore.ParameterSet.Config as cms

process = cms.Process("TRA")


#
# This runs over a DIGI file (that contains clusters and stubs)
# and runs the tracklet-based L1Tracking.
# Note that the tracklet-L1Tracks are already present on the
# DIGI produced in SLHC8 - but you may want to rerun it, to
# benefit from the latest updates of the L1Tracking code.
# 
# For running over a SLHC6 DIGI file: the file must contain the
# tracker digis (they are present by default in SLHC6 DIGI files).
# If you run over the centrally produced files, you need to redo
# the stubs, see below and uncomment the corresponding lines.
#


process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )


process.source = cms.Source("PoolSource",
    #fileNames = singleElectronFiles
    #fileNames = ttbarFiles_p1
    #fileNames = cms.untracked.vstring('/store/group/comm_trigger/L1TrackTrigger/BE5D_620_SLHC6/singleMu/PU140/SingleMuMinus_BE5D_PU140.root')
     fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/e/eperez/public/step2.root')
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')


# ---------------------------------------------------------------------------
#
# ---- Run the L1Tracking :

# ---- redo the stubs in case you run over a 620_SLHC6 file.
#      Stubs were produced during the central production
#      and are present on the DIGI files, but the "z-matching" condition
#      was enforced. Here we redo the stubs without the z-matching.
#      This leads to better tracking efficiencies.
 
#process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
#process.pStubs = cms.Path( process.L1TkStubsFromPixelDigis )

# --- now we run the L1Track producer :

process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')

process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

process.BeamSpotFromSim =cms.EDProducer("BeamSpotFromSimProducer")

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TTrack_cfi")
process.L1Tracks.geometry = cms.untracked.string('BE5D')

process.pL1Tracks = cms.Path( process.BeamSpotFromSim*process.L1Tracks )

#
# ---------------------------------------------------------------------------


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_w_Tracks.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)

process.Out.outputCommands.append( 'keep *_*_*_TRA' )
process.Out.outputCommands.append('keep *_generator_*_*')
process.Out.outputCommands.append('keep *_*gen*_*_*')
process.Out.outputCommands.append('keep *_*Gen*_*_*')
process.Out.outputCommands.append('keep *_rawDataCollector_*_*')

process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')

process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')

process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_*')



process.FEVToutput_step = cms.EndPath(process.Out)




