import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#
# This runs over a file that already contains the L1Tracks.
#
#
# It also runs a trivial analyzer than prints the objects
# that have been created. 


#from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *
from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p2_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.singleMuonFiles_cfi import *

process.source = cms.Source("PoolSource",
     fileNames = minBiasFiles_p2
     #fileNames = minBiasFiles_p1
     #fileNames = singleMuonFiles
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')




# ---------------------------------------------------------------------------
#
# --- Produces the Run-1 L1muon objects 
#

process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')


# --- creates l1extra objects for L1Muons 
        
        # raw2digi to get the GT digis
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.p0 = cms.Path( process.RawToDigi )
        # run L1Reco
process.load('Configuration.StandardSequences.L1Reco_cff')
process.L1Reco = cms.Path( process.l1extraParticles )


# ---------------------------------------------------------------------------


# 
# ---  Produces the L1TkMuons from the naive DeltaR producer ::
#

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkMuonProducer_cfi")
process.pMuonsNaive = cms.Path( process.L1TkMuonsNaive )



#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

#
# --- The central muons from the Padova algorithm
#


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


process.L1TkMuonsDT = cms.EDProducer("L1TkMuonDTProducer"
)

process.pMuonsDT = cms.Path( process.L1TkMuonsDT )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

#
# --- The muons from Slava

process.load("L1Trigger.L1ExtraFromDigis.l1extraMuExtended_cfi")
process.load("SLHCUpgradeSimulations.L1TrackTrigger.l1TkMuonsExt_cff")

# this is based on all GMTs available (BX=0 is hardcoded )                                                                               
process.l1tkMusFromExtendedAllEta =  cms.Path(process.l1extraMuExtended * process.l1TkMuonsExt )

# this is based on CSCTF record directly (no GMT sorting) and creates TkMus in |eta| > 1.1                         
process.l1tkMusFromExtendedForward = cms.Path(process.l1extraMuExtended * process.l1TkMuonsExtCSC )



# ---------------------------------------------------------------------------

#
# finally, merge the collections of TkMuons.
# 	- for the central region: take the Padova's algorithm
#	- for non-central, take Slava's algorithm
#		- at high eta, CSC + new chambers are used 
#		- in the intermediate region, GMT muons
#		  are used.
# Currently waiting for Slava's code. So I use the
# TkMuons created by the naive producer.
#

process.L1TkMuonsMerge = cms.EDProducer("L1TkMuonMerger",
   TkMuonCollections = cms.VInputTag( cms.InputTag("L1TkMuonsDT","DTMatchInwardsTTTrackFullReso"), 
                                      #cms.InputTag("L1TkMuonsNaive",""), 
			  	      cms.InputTag("l1TkMuonsExt",""),
				      cms.InputTag("l1TkMuonsExtCSC","") ),
   absEtaMin = cms.vdouble( 0., 0. , 1.1),	# Padova's not ready yet
   absEtaMax = cms.vdouble( 0., 1.1 , 5.0) 
)

process.pMuonsMerge = cms.Path( process.L1TkMuonsMerge) 


# Run a trivial analyzer that prints the objects

process.ana = cms.EDAnalyzer( 'PrintL1TkObjects' ,
    L1VtxInputTag = cms.InputTag("L1TkPrimaryVertex"),		# dummy here
    L1TkEtMissInputTag = cms.InputTag("L1TkEtMiss","MET"),	# dummy here
    L1TkElectronsInputTag = cms.InputTag("L1TkElectrons","EG"), # dummy here
    L1TkPhotonsInputTag = cms.InputTag("L1TkPhotons","EG"),	# dummy here
    L1TkJetsInputTag = cms.InputTag("L1TkJets","Central"),	# dummy here
    L1TkHTMInputTag = cms.InputTag("L1TkHTMissCaloHI","")	# dummy here
)

#process.pAna = cms.Path( process.ana )



# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_muons.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


process.Out.outputCommands.append('keep *_L1TkMuons*_*_*')
process.Out.outputCommands.append('keep *_l1extraMuExtended_*_*')
process.Out.outputCommands.append('keep *_l1TkMuonsExt*_*_*')

process.Out.outputCommands.append('keep *L1Muon*_l1extraParticles_*_*')

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







