import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

#
# This runs over a file that already contains the L1Tracks.
#
# It produces the following objects :
#    - L1EG objects obtained by running the SLHCCaloTrigger sequence
#	- this produces both the "old stage-2" and the "new stage-2" objects
#    - collection of L1TkEmParticles  - produces Trk-based isolated "photons"
#    - collection of L1TkElectrons from L1TkElectronTrackProducer 
#
#
# It also runs a trivial analyzer than prints the objects
# that have been created. 


process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
         '/store/group/comm_trigger/L1TrackTrigger/620_SLHC8/BE5D/SingleEle_NoPU.root'
     )
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
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

process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')

process.load('Configuration/StandardSequences/L1HwVal_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")


	# The sequence SLHCCaloTrigger is broken in SLHC8 because of the
	# L1Jets stuff (should be fixed in SLHC9).
	# Hence, in the "remove" lines below, I hack the sequence such that
	# we can run the L1EGamma stuff.

process.SLHCCaloTrigger.remove( process.L1TowerJetProducer )
process.SLHCCaloTrigger.remove( process.L1TowerJetFilter1D )
process.SLHCCaloTrigger.remove( process.L1TowerJetFilter2D )
process.SLHCCaloTrigger.remove( process.L1TowerJetPUEstimator )
process.SLHCCaloTrigger.remove( process.L1TowerJetPUSubtractedProducer )
process.SLHCCaloTrigger.remove( process.L1CalibFilterTowerJetProducer )
process.SLHCCaloTrigger.remove( process.L1CaloJetExpander )

# bug fix for missing HCAL TPs in MC RAW
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import HcalTPGCoderULUT
HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)
process.valRctDigis.hcalDigis             = cms.VInputTag(cms.InputTag('valHcalTriggerPrimitiveDigis'))
process.L1CaloTowerProducer.HCALDigis =  cms.InputTag("valHcalTriggerPrimitiveDigis")

process.slhccalo = cms.Path( process.RawToDigi + process.valHcalTriggerPrimitiveDigis+process.SLHCCaloTrigger)


	# To produce L1EG objects corresponding
	# to the Run-1 L1EG algorithm, one just needs to run
	# L1Reco. The (Run-1) L1EG algorithm has already been
	# run in the DIGI step of the production. 
process.load('Configuration.StandardSequences.L1Reco_cff')
process.L1Reco = cms.Path( process.l1extraParticles )

#
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------

# Now we produce L1TkEmParticles and L1TkElectrons


# ----  "photons" isolated w.r.t. L1Tracks :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEmParticleProducer_cfi")
process.pL1TkPhotons = cms.Path( process.L1TkPhotons )

# ----  "electrons" from L1Tracks. Inclusive electrons :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkElectronTrackProducer_cfi")
process.pElectrons = cms.Path( process.L1TkElectrons )

# ----  L1TkElectrons that are isolated w.r.t. L1Tracks :

process.L1TkIsoElectrons = process.L1TkElectrons.clone()
process.L1TkIsoElectrons.IsoCut = cms.double(0.1)
process.pElectronsTkIso = cms.Path( process.L1TkIsoElectrons )

# ----  L1TkElectrons made from L1IsoEG objects, i.e. from L1EG objects that 
#       are isolated in the calorimeter :

process.L1TkElectronsIsoEG = process.L1TkElectrons.clone()
process.L1TkElectronsIsoEG.L1EGammaInputTag = cms.InputTag("SLHCL1ExtraParticlesNewClustering","IsoEGamma")
process.pElectronsIsoEG = cms.Path( process.L1TkElectronsIsoEG )

# ----  "electrons" from stubs  	- not implemented yet.
#                                               
#process.L1TkElectronsStubs = cms.EDProducer("L1TkElectronStubsProducer",
#)                                              
#process.pElectronsStubs = cms.Path( process.L1TkElectronsStubs )



# ---------------------------------------------------------------------------


# Run a trivial analyzer that prints the objects

process.ana = cms.EDAnalyzer( 'L1TrackTriggerObjectsAnalyzer' ,
    L1VtxInputTag = cms.InputTag("L1TkPrimaryVertex"),
    L1TkEtMissInputTag = cms.InputTag("L1TkEtMiss","MET"),
    L1TkElectronsInputTag = cms.InputTag("L1TkElectronsTrack","EG"),
    L1TkPhotonsInputTag = cms.InputTag("L1TkPhotons","EG"),
    L1TkJetsInputTag = cms.InputTag("L1TkJets","Central"),
    L1TkHTMInputTag = cms.InputTag("L1TkHTMiss")
)

process.pAna = cms.Path( process.ana )



# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "test_L1TrackTriggerObjects.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


	# gen-level information
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

	# the L1EG objects
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_*_*' )
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticlesNewClustering_*_*')
process.Out.outputCommands.append('keep l1extraL1Em*_l1extraParticles_*_*')

	# the L1TkEmParticles
process.Out.outputCommands.append('keep *_L1TkPhotons_*_*')

	# the L1TkElectrons
process.Out.outputCommands.append('keep *_L1TkElectrons_*_*')
process.Out.outputCommands.append('keep *_L1TkElectronsIsoEG_*_*')
process.Out.outputCommands.append('keep *_L1TkIsoElectrons_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)







