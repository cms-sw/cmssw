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
#    '/store/group/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_1.root',
     ##
     ## rate test sample:
     ## 
      #'/store/mc/TTI2023Upg14D/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/022FFF01-E4E0-E311-9DAD-002618943919.root',
      #'/store/mc/TTI2023Upg14D/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/0257B6AB-00E6-E311-A12F-0025905A6068.root',
## VBF H->tautau
      '/store/mc/TTI2023Upg14D/VBF_HToTauTau_125_14TeV_powheg_pythia6/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v1/00000/00114910-0DE9-E311-B42B-0025905A60F4.root',
      '/store/mc/TTI2023Upg14D/VBF_HToTauTau_125_14TeV_powheg_pythia6/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v1/00000/06B547C1-03E9-E311-864E-0025905A48D6.root',
      '/store/mc/TTI2023Upg14D/VBF_HToTauTau_125_14TeV_powheg_pythia6/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v1/00000/0EFF9054-0CE9-E311-94C0-0025905A60F4.root',
      '/store/mc/TTI2023Upg14D/VBF_HToTauTau_125_14TeV_powheg_pythia6/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v1/00000/16C00111-F8E8-E311-9E15-0025905A60F4.root',
      '/store/mc/TTI2023Upg14D/VBF_HToTauTau_125_14TeV_powheg_pythia6/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v1/00000/2E87B579-0BE9-E311-AB95-0025905A60E0.root',
     )
)

# --- root output
process.TFileService = cms.Service("TFileService", fileName = cms.string('TkEmTau.root'), closeFileFast = cms.untracked.bool(True))


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')



# ---------------------------------------------------------------------------
#
# --- Recreate the L1Tracks to benefit from the latest updates

process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TrackingSequence_cfi")
process.pTracking = cms.Path( process.DefaultTrackingSequence )


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
#
# --- Produces the L1calo objects 
#
	# To produce L1EG objects corresponding to the "stage-2" algorithms:
	# one runs the SLHCCaloTrigger sequence. This produces both the
	# "old stage-2" objects (2x2 clustering) and the "new stage-2"
	# objects (new clustering from JB Sauvan et al). Note that the
	# efficiency of the latter is currently poor at very high PU.

# The sequence SLHCCaloTrigger creates "stage-2" L1Taus.
# Two collections are created:
# a) ("SLHCL1ExtraParticles","Taus")
# b) ("SLHCL1ExtraParticles","IsoTaus")
# So far only the ("SLHCL1ExtraParticles","Taus") collection has been used.
# The ("SLHCL1ExtraParticles","IsoTaus") has not been looked yet.

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
   EtminForStore = cms.double(-1.0),                                                
   debug = cms.untracked.bool(False),
   useECalEndcap = cms.bool(True)
)
process.pSasha = cms.Path( process.L1EGammaCrystalsProducer )

# --------------------------------------------------------------------------------------------


# ---------------------------------------- stage - 2 calibrated taus -------------------

# Produce calibrated (eT-corrected) L1CaloTaus:
process.L1CaloTauCorrectionsProducer = cms.EDProducer("L1CaloTauCorrectionsProducer", 
     L1TausInputTag = cms.InputTag("SLHCL1ExtraParticles","Taus") 
)

# --------------------------------------------------------------------------------


# Setup the L1TkTauFromCalo producer:
process.L1TkTauFromCaloProducer = cms.EDProducer("L1TkTauFromCaloProducer",
      #L1TausInputTag                   = cms.InputTag("SLHCL1ExtraParticles","Taus"),
      L1TausInputTag                   = cms.InputTag("L1CaloTauCorrectionsProducer","CalibratedTaus"),
      L1TrackInputTag                  = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
      L1TkTrack_ApplyVtxIso            = cms.bool( True  ),      # Produce vertex-isolated L1TkTaus?
      L1TkTrack_VtxIsoZ0Max            = cms.double( 1.0  ),     # Max vertex z for L1TkTracks for VtxIsolation [cm]
      L1TkTrack_NStubsMin              = cms.uint32(  5   ),     # Min number of stubs per L1TkTrack [unitless]
      L1TkTrack_PtMin_AllTracks        = cms.double(  2.0 ),     # Min pT applied on all L1TkTracks [GeV]
      L1TkTrack_PtMin_SignalTracks     = cms.double(  10.0),     # Min pT applied on signal L1TkTracks [GeV]
      L1TkTrack_PtMin_IsoTracks        = cms.double(  2.0 ),     # Min pT applied on isolation L1TkTracks [GeV]
      L1TkTrack_RedChiSquareEndcapMax  = cms.double(  5.0 ),     # Max red-chi squared for L1TkTracks in Endcap
      L1TkTrack_RedChiSquareBarrelMax  = cms.double(  2.0 ),     # Max red-chi squared for L1TkTracks in Barrel
      L1TkTrack_VtxZ0Max               = cms.double( 30.0 ),     # Max vertex z for L1TkTracks [cm] 
      DeltaR_L1TkTau_L1TkTrack         = cms.double( 0.10 ),     # Cone size for L1TkTracks assigned to L1TkTau
      DeltaR_L1TkTau_Isolation_Min     = cms.double( 0.10 ),     # Isolation-cone size (min) - becomes isolation annulus if > 0.0
      DeltaR_L1TkTau_Isolation_Max     = cms.double( 0.40 ),     # Isolation cone size (max)
      DeltaR_L1TkTau_L1CaloTau         = cms.double( 0.15 ),     # Matching cone for L1TkTau and L1CaloTau
      L1CaloTau_EtMin                  = cms.double( 5.0  ),     # Min eT applied on all L1CaloTaus [GeV]
      RemoveL1TkTauTracksFromIsoCalculation = cms.bool( False ), # Remove tracks used in L1TkTau construction from VtxIso calculation?
)

process.pCorr = cms.Path( process.L1CaloTauCorrectionsProducer )
process.caloTaus = cms.Path( process.L1TkTauFromCaloProducer )


# Setup the L1CaloTau producer:
process.L1CaloTauProducer = cms.EDProducer("L1CaloTausToTkTausTranslator",
      L1TausInputTag                   = cms.InputTag("SLHCL1ExtraParticles","Taus"),
      #L1TausInputTag                   = cms.InputTag("L1CaloTauCorrectionsProducer","CalibratedTaus"),
)

process.plaincaloTaus = cms.Path( process.L1CaloTauProducer )


# -------------- Tracker only based taus ------------------------------------

process.L1TkTauFromL1Track = cms.EDProducer("L1TkTauFromL1TrackProducer",
                                            L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
                                            ZMAX = cms.double( 25. ),# in cm
                                            CHI2MAX = cms.double( 100. ),
                                            PTMINTRA = cms.double( 2. ),# in GeV
                                            DRmax = cms.double( 0.5 ),
                                            nStubsmin = cms.int32( 5 )        # minimum number of stubs
                                            )

process.tkOnlyTaus = cms.Path( process.L1TkTauFromL1Track )



# ---------------------------------------------------------------------------


# Run a trivial analyzer that prints the objects

#process.ana = cms.EDAnalyzer( 'PrintL1TkObjects' ,
#    L1VtxInputTag = cms.InputTag("L1TkPrimaryVertex"),		# dummy here
#    L1TkMuonsInputTag = cms.InputTag("L1TkMuons"),		# dummy here
#    L1TkEtMissInputTag = cms.InputTag("L1TkEtMiss","MET"),	# dummy here
#    L1TkElectronsInputTag = cms.InputTag("L1TkElectrons","EG"),
#    L1TkPhotonsInputTag = cms.InputTag("L1TkPhotons","EG"),
#    L1TkJetsInputTag = cms.InputTag("L1TkJets","Central"),	# dummy here
#    L1TkHTMInputTag = cms.InputTag("L1TkHTMissCaloHI",""),	# dummy here
#    L1EmInputTag = cms.InputTag("l1ExtraCrystalProducer","EGammaCrystal")      
#)
#
#process.pAna = cms.Path( process.ana )

process.tau = cms.EDAnalyzer( 'L1TkEmTauNtupleMaker' ,
    MyProcess = cms.int32(13),
    DebugMode = cms.bool(False),                           
    L1EmInputTag = cms.InputTag("L1EGammaCrystalsProducer","EGCrystalCluster")      
)

process.pTau = cms.Path( process.tau )


process.tkemtau = cms.EDProducer( 'L1TkEmTauProducer' ,       
                                  L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
                                  L1EmInputTag = cms.InputTag("L1EGammaCrystalsProducer","EGCrystalCluster"),
                                  ptleadcut = cms.double(5.0),
                                  ptleadcone = cms.double(0.3),
                                  masscut = cms.double(1.77),
                                  emptcut = cms.double(5.0),
                                  trketacut = cms.double(2.3),
                                  pttotcut = cms.double(5.0),
                                  isocone = cms.double(0.25),
                                  isodz = cms.double(0.6),
                                  relisocut = cms.double(0.15),
                                  chisqcut = cms.double(40.0),
                                  nstubcut = cms.int32(5),
                                  dzcut = cms.double(0.8)                                  
)

process.pTkemtau = cms.Path( process.tkemtau )


process.tkemtauAnalyzer = cms.EDAnalyzer( 'L1TrackTauAnalyzer' ,
    L1TkTauInputTag = cms.InputTag("tkemtau","")      
)

process.ptkemtauAnalyzer = cms.Path( process.tkemtauAnalyzer )

process.L1TkTauFromCaloProducerAnalyzer = cms.EDAnalyzer( 'L1TrackTauAnalyzer' ,
    L1TkTauInputTag = cms.InputTag("L1TkTauFromCaloProducer","")      
)

process.pL1TkTauFromCaloProducerAnalyzer = cms.Path( process.L1TkTauFromCaloProducerAnalyzer )

process.L1TkTauFromL1TrackAnalyzer = cms.EDAnalyzer( 'L1TrackTauAnalyzer' ,
    L1TkTauInputTag = cms.InputTag("L1TkTauFromL1Track","")      
)

process.pL1TkTauFromL1TrackAnalyzer = cms.Path( process.L1TkTauFromL1TrackAnalyzer )


process.L1CaloTauProducerAnalyzer = cms.EDAnalyzer( 'L1TrackTauAnalyzer' ,
    L1TkTauInputTag = cms.InputTag("L1CaloTauProducer","")      
)

process.pL1CaloTauProducerAnalyzer = cms.Path( process.L1CaloTauProducerAnalyzer )




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
#process.Out.outputCommands.append('keep *_genParticles_*_*')


	# the L1Tracks, clusters and stubs
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
#process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')
#process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')
#process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')
#process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_*')

	# the L1EG objects
#process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_*_*' )
#process.Out.outputCommands.append('keep *_SLHCL1ExtraParticlesNewClustering_*_*')
#process.Out.outputCommands.append('keep l1extraL1Em*_l1extraParticles_*_*')
		# for crystal-level granularity :
#process.Out.outputCommands.append('keep *_L1EGammaCrystalsProducer_*_*')
#process.Out.outputCommands.append('keep *_l1ExtraCrystalProducer_*_*')

	# the L1TkEmParticles
#process.Out.outputCommands.append('keep *_L1TkPhotons_*_*')

	# the L1TkElectrons
#process.Out.outputCommands.append('keep *_L1TkElectrons_*_*')
#process.Out.outputCommands.append('keep *_L1TkElectronsCrystal_*_*')   # for crystal-level granularity
#process.Out.outputCommands.append('keep *_L1TkElectronsIsoEG_*_*')
#process.Out.outputCommands.append('keep *_L1TkIsoElectrons_*_*')

#process.Out.outputCommands.append('keep *_L1TkElectronsLoose_*_*')
#process.Out.outputCommands.append('keep *_L1TkIsoElectronsLoose_*_*')
#process.Out.outputCommands.append('keep *_L1TkElectronsLooseV2_*_*')


# --- to use the genParticles, one needs to keep the collections of associators below:
#process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
#process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
#process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')

#process.Out.outputCommands.append('keep *_*_*_*' )
#process.FEVToutput_step = cms.EndPath(process.Out)







