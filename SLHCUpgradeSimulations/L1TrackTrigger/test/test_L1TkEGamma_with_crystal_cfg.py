import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

#
# This first creates the collection of "L1Tracks for electrons" by running
# the FullTrackingSequence.
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


# to run over the test rate sample (part 1) :
#from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *
#process.source = cms.Source("PoolSource",
#     fileNames = minBiasFiles_p1
#)

# to run over another sample:
process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
     # electron file:
  '/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/Electrons/PU140/SingleElectron_E2023TTI_PU140_9.root'
     )
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')


# ---------------------------------------------------------------------------
#
# --- Create the collection of special tracks for electrons
#

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TrackingSequence_cfi")
process.pTracking = cms.Path( process.ElectronTrackingSequence )

# ---------------------------------------------------------------------------


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

process.load('Configuration/StandardSequences/L1HwVal_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_forTTI_cff")

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


# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters (code of Sasha Savin & Nick Smith)

        # first you need the ECAL RecHIts :
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.reconstruction_step = cms.Path( process.calolocalreco )

process.L1EGammaCrystalsProducer = cms.EDProducer("L1EGCrystalClusterProducer",
   EtminForStore = cms.double( -999.),
   debug = cms.untracked.bool(False),
   useECalEndcap = cms.bool(True)
)
process.pSasha = cms.Path( process.L1EGammaCrystalsProducer )

# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
#
# ----  Match the L1EG stage-2 objects created by the SLHCCaloTrigger sequence above
#       with the crystal-level clusters.
#       This produces a new collection of L1EG objects, starting from the original
#       L1EG collection. The eta and phi of the L1EG objects is corrected using the
#       information of the xtal level clusters.

# ---- Note: this is not needed anymore. The latest code of L1EGCrystalClusterProducer,
#      provides similar rates as the new stage-2 algorithm (and with a better
#      efficiency). Hence we don;t need anymore to rely on the tower-based
#      new stage-2 algorithm for the ele-ID of the EGamma object, we can
#      directly use teh clusters from Sasha & Nick.

#process.l1ExtraCrystalProducer = cms.EDProducer("L1ExtraCrystalPosition",
#   eGammaSrc = cms.InputTag("SLHCL1ExtraParticlesNewClustering","EGamma"),
#   eClusterSrc = cms.InputTag("L1EGammaCrystalsProducer","EGCrystalCluster")
#)
#process.egcrystal_producer = cms.Path(process.l1ExtraCrystalProducer)




# ---------------------------------------------------------------------------

# Now we produce L1TkEmParticles and L1TkElectrons


# ----  "electrons" from L1Tracks. Inclusive electrons :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkElectronTrackProducer_cfi")
process.pElectrons = cms.Path( process.L1TkElectrons )

	# --- collection of L1TkElectrons made from "crystal-level" L1EG objects
process.L1TkElectronsCrystal = process.L1TkElectrons.clone()
process.L1TkElectronsCrystal.L1EGammaInputTag = cms.InputTag("L1EGammaCrystalsProducer","EGammaCrystal")
	# ... of course, the cuts need to be retuned for the xtal-level granularity
process.pElectronsCrystal = cms.Path( process.L1TkElectronsCrystal )



# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


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

	# the L1EG objects
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_*_*' )
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticlesNewClustering_*_*')
process.Out.outputCommands.append('keep l1extraL1Em*_l1extraParticles_*_*')
		# for crystal-level granularity :
process.Out.outputCommands.append('keep *_L1EGammaCrystalsProducer_*_*')
process.Out.outputCommands.append('keep *_l1ExtraCrystalProducer_*_*')


	# the L1TkElectrons
process.Out.outputCommands.append('keep *_L1TkElectrons_*_*')
process.Out.outputCommands.append('keep *_L1TkElectronsCrystal_*_*')   # for crystal-level granularity


# --- to browse the genParticles in ROOT, one needs to keep the collections of associators below:
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)







