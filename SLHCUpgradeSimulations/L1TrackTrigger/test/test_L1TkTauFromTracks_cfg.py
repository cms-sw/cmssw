import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#
# This runs over a file that already contains the L1Tracks.
#


# to run over the test rate sample (part 1) :
from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *
process.source = cms.Source("PoolSource",
     fileNames = minBiasFiles_p1
)

#from SLHCUpgradeSimulations.L1TrackTrigger.singleTau1pFiles_cfi import *
#process.source = cms.Source("PoolSource",
#     fileNames = singleTau1pFiles
#)


# to run over another sample:
#process.source = cms.Source("PoolSource",
#'/store/mc/TTI2023Upg14D/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/022FFF01-E4E0-E311-9DAD-002618943919.root',
     #)
#)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')




# ---------------------------------------------------------------------------
#
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')

process.L1TkTauFromL1Track = cms.EDProducer("L1TkTauFromL1TrackProducer",
                                            L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
                                            ZMAX = cms.double( 25. ),# in cm
                                            CHI2MAX = cms.double( 100. ),
                                            PTMINTRA = cms.double( 2. ),# in GeV
                                            DRmax = cms.double( 0.5 ),
                                            nStubsmin = cms.int32( 5 )        # minimum number of stubs
                                            )

process.pTaus = cms.Path( process.L1TkTauFromL1Track )


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# Run a trivial analyzer that prints the objects

process.ana = cms.EDAnalyzer( 'PrintL1TkObjects' ,
    L1VtxInputTag = cms.InputTag("L1TkPrimaryVertex"),		# dummy here
    L1TkEtMissInputTag = cms.InputTag("L1TkEtMiss","MET"),	# dummy here
    L1TkElectronsInputTag = cms.InputTag("L1TkElectrons","EG"),
    L1TkPhotonsInputTag = cms.InputTag("L1TkPhotons","EG"),
    L1TkJetsInputTag = cms.InputTag("L1TkJets","Central"),	# dummy here
    L1TkHTMInputTag = cms.InputTag("L1TkHTMissCaloHI",""),	# dummy here
    L1TkMuonsInputTag = cms.InputTag("L1TkMuons","")		# dummy here

)

#process.pAna = cms.Path( process.ana )



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

	# the L1TkTaus
process.Out.outputCommands.append('keep *_L1TkTauFromL1Track_*_*')


# --- to browse the genParticles, one needs to keep the collections of associators below:
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)







