import FWCore.ParameterSet.Config as cms

process = cms.Process("Muo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)

#
# This runs over a file that already contains the L1Tracks.
#
# Creates L1TkMuons starting from the Run-1 L1Muons.
#

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
     fileNames = cms.untracked.vstring(
	'/store/group/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/Muons/PU140/SingleMuon_E2023TTI_PU140_55.root'
     )
)


process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')



# --- creates l1extra objects for L1Muons 

        # raw2digi to get the GT digis
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.p0 = cms.Path( process.RawToDigi )
	# run L1Reco
process.load('Configuration.StandardSequences.L1Reco_cff')
process.L1Reco = cms.Path( process.l1extraParticles )


# ---  Produces the L1TkMuons :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkMuonProducer_cfi")
process.pMuons = cms.Path( process.L1TkMuons )



# ---------------------------------------------------------------------------


# Run a trivial analyzer that prints the objects

process.ana = cms.EDAnalyzer( 'PrintL1TkObjects' ,
    L1VtxInputTag = cms.InputTag("L1TkPrimaryVertex"),          # dummy here
    L1TkEtMissInputTag = cms.InputTag("L1TkEtMiss","MET"),      # dummy here
    L1TkElectronsInputTag = cms.InputTag("L1TkElectrons","EG"), # dummy here
    L1TkPhotonsInputTag = cms.InputTag("L1TkPhotons","EG"),     # dummy here
    L1TkJetsInputTag = cms.InputTag("L1TkJets","Central"),      # dummy here
    L1TkHTMInputTag = cms.InputTag("L1TkHTMissCaloHI",""),      # dummy here
    L1TkMuonsInputTag = cms.InputTag("L1TkMuons","")            

)

process.pAna = cms.Path( process.ana )





# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "test_L1TrackTriggerObjects.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


        # genlevel information
#process.Out.outputCommands.append('keep *_generator_*_*')
#process.Out.outputCommands.append('keep *_*gen*_*_*')
#process.Out.outputCommands.append('keep *_*Gen*_*_*')
#process.Out.outputCommands.append('keep *_rawDataCollector_*_*')

        # the L1Tracks, clusters and stubs
process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_*')
process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_*')
process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_*')

        # the L1TkMuon
process.Out.outputCommands.append('keep *_L1TkMuons_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)

