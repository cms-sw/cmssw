import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

#
# This runs over a file that already contains the L1Tracks.
#
# It produces the following objects :
#    - L1TkPrimaryVertex  - note that the clusters and stubs must
#	   have been kept on the event. See the event content of
#	   the example runL1Tracks.py
#    - collection of L1TkEmParticles  - produces Trk-based isolated "photons"
#    - collection of L1TkElectrons from L1TkElectronTrackProducer 
#    - L1TkEtMiss
#    - collection of L1TkMuons
 
# This runs the L1EG algorithms (stage-2 and new clustering), and 
# creates L1TkEmParticles, L1TkElectrons, L1TrkMET.
#
# It also unpacks the L1Jets (Run-1 algorithm) that have been created
# during the centrakl production. They are used to create L1TkJets and
# L1TkHTMiss - these are just technical templates so far.
#


process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/e/eperez/public/step2.root')
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')



# --- Produce the Primary Vertex

process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkPrimaryVertexProducer_cfi")
process.pL1TkPrimaryVertex = cms.Path( process.L1TkPrimaryVertex )

# --- Produce the L1TrkMET
process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEtMissProducer_cfi")
process.pL1TrkMET = cms.Path( process.L1TkEtMiss )



# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "test_L1TrackTriggerObjects.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


	# genlevel information
process.Out.outputCommands.append('keep *_generator_*_*')
process.Out.outputCommands.append('keep *_*gen*_*_*')
process.Out.outputCommands.append('keep *_*Gen*_*_*')
process.Out.outputCommands.append('keep *_rawDataCollector_*_*')

        # the L1Tracks, clusters and stubs
process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')
process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_*')

	# the L1TkPrimaryVertex
process.Out.outputCommands.append('keep *_L1TkPrimaryVertex_*_*')

	# the TrkMET
process.Out.outputCommands.append('keep *_L1TkEtMiss_*_*')



process.FEVToutput_step = cms.EndPath(process.Out)







