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


process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
     # single muons
     #'/store/group/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/Muons/PU140/SingleMuon_E2023TTI_PU140_53.root'
     '/store/group/comm_trigger/L1TrackTrigger/620_SLHC10/Extended2023TTI/Muons/PU140/m1_SingleMuon_E2023TTI_PU140.root'
     )
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'POSTLS261_V3::All', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')




# ---------------------------------------------------------------------------
#
# --- Produces the L1EG objects 
#
	# To produce L1EG objects corresponding to the "stage-2" algorithms:
	# one runs the SLHCCaloTrigger sequence. This produces both the
	# "old stage-2" objects (2x2 clustering) and the "new stage-2"
	# objects (new clustering from JB Sauvan et al). Note that the
	# efficiency of the latter is currently poor at very high PU.

process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')


#process.load('Configuration/StandardSequences/L1HwVal_cff')
#process.load('Configuration.StandardSequences.RawToDigi_cff')
#process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

#process.load('Configuration.StandardSequences.L1Reco_cff')
#process.L1Reco = cms.Path( process.l1extraParticles )

#
# ---------------------------------------------------------------------------


#################################################################################################
# now, all the DT related stuff
#################################################################################################
# to produce, in case, collection of L1MuDTTrack objects:
#process.dttfDigis = cms.Path(process.simDttfDigis)

# the DT geometry
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("SimMuon/DTDigitizer/muonDTDigis_cfi")
##process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")

#################################################################################################
# define the producer of DT + TK objects
#################################################################################################
process.load("L1Trigger.DTPlusTrackTrigger.DTPlusTrackProducer_cfi")
process.DTPlusTk_step = cms.Path(process.DTPlusTrackProducer)






# ---------------------------------------------------------------------------

# Now we produce L1TkEmParticles and L1TkElectrons

process.L1TkMuons = cms.EDProducer("L1TkMuonParticleProducer"
)

process.pMuons = cms.Path( process.L1TkMuons )

# ---------------------------------------------------------------------------


# Run a trivial analyzer that prints the objects

process.ana = cms.EDAnalyzer( 'PrintL1TkObjects' ,
    L1VtxInputTag = cms.InputTag("L1TkPrimaryVertex"),		# dummy here
    L1TkEtMissInputTag = cms.InputTag("L1TkEtMiss","MET"),	# dummy here
    L1TkElectronsInputTag = cms.InputTag("L1TkElectrons","EG"),
    L1TkPhotonsInputTag = cms.InputTag("L1TkPhotons","EG"),
    L1TkJetsInputTag = cms.InputTag("L1TkJets","Central"),	# dummy here
    L1TkHTMInputTag = cms.InputTag("L1TkHTMissCaloHI","")	# dummy here
)

#process.pAna = cms.Path( process.ana )



# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


process.Out.outputCommands.append('keep *_L1TkMuons_*_*')
#process.Out.outputCommands.append('keep *_DTPlusTrackProducer_*_*')

	# raw data
#process.Out.outputCommands.append('keep *_rawDataCollector_*_*')


	# gen-level information
#process.Out.outputCommands.append('keep *_generator_*_*')
#process.Out.outputCommands.append('keep *_*gen*_*_*')
#process.Out.outputCommands.append('keep *_*Gen*_*_*')
process.Out.outputCommands.append('keep *_genParticles_*_*')


	# the L1Tracks, clusters and stubs
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
#process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')
#process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')
#process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')
#process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_*')

# --- to use the genParticles, one needs to keep the collections of associators below:
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)







