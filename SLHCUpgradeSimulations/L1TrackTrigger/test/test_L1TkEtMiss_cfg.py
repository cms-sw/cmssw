import FWCore.ParameterSet.Config as cms

process = cms.Process("MET")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#
# This runs over a file that already contains the L1Tracks.
#
# This runs over a file containing L1Tracks.
# It produces the following objects :
#    - L1TkPrimaryVertex  - running the L1TkFastVertexProducer
#    - L1TkEtMiss
#
# In the end, it also retrieves the calo-based L1MET corresponding
# to the Run-1 algorithm. For this, the file must contain the rawData
# or at least the GCT raw data.
# 

#from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *
from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p2_cfi import *


process.source = cms.Source("PoolSource",
# rate test sample (private production)
    #fileNames = minBiasFiles_p1
    fileNames = minBiasFiles_p2
# full sample from central production is in store/mc :
    #fileNames = cms.untracked.vstring(
#'/store/mc/TTI2023Upg14D/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/022FFF01-E4E0-E311-9DAD-002618943919.root'
    #)
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')



# --- Produce the Primary Vertex

process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkPrimaryVertexProducer_cfi")
process.pL1TkPrimaryVertex = cms.Path( process.L1TkPrimaryVertex )

# --- Produce the L1TrkMET

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEtMissProducer_cfi")
process.pL1TrkMET = cms.Path( process.L1TkEtMiss )

#process.L1TkEtMiss2 = process.L1TkEtMiss.clone()
#process.L1TkEtMiss2.DeltaZ = cms.double(0.2)
#process.pL1TrkMET2 = cms.Path( process.L1TkEtMiss2 )

#process.L1TkEtMiss3 = process.L1TkEtMiss.clone();
#process.L1TkEtMiss3.PTMAX = cms.double(-1)
#process.pL1TrkMET3 = cms.Path( process.L1TkEtMiss3 )

#process.L1TkEtMiss4 = process.L1TkEtMiss.clone()
#process.L1TkEtMiss4.nStubsPSmin = cms.int32( 3 )
#process.pL1TrkMET4 = cms.Path( process.L1TkEtMiss4 )

#process.L1TkEtMiss5 = process.L1TkEtMiss.clone()
#process.L1TkEtMiss5.HighPtTracks = cms.int32( 1 )
#process.pL1TrkMET5 = cms.Path( process.L1TkEtMiss5 )

#process.L1TkEtMiss6 = process.L1TkEtMiss.clone()
#process.L1TkEtMiss6.PTMAX = cms.double(999999999.)
#process.pL1TrkMET6 = cms.Path( process.L1TkEtMiss6 )



# --- Retrieve the Calo-based L1MET corresponding to the Run-1 algorithm:

	# first, need to get the GCT digis
#process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('L1Trigger.Configuration.L1RawToDigi_cff')
process.p0 = cms.Path( process.L1RawToDigi )

	# second, run L1Reco to produce the L1ETM object corresponding
        # to the current trigger
process.load('Configuration.StandardSequences.L1Reco_cff')
process.L1Reco = cms.Path( process.l1extraParticles )




# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)



        # the L1TkPrimaryVertex
process.Out.outputCommands.append('keep *_L1TkPrimaryVertex_*_*')

        # the TrkMET
process.Out.outputCommands.append('keep *_L1TkEtMiss*_*_*')

	# the calo-based L1MET 
process.Out.outputCommands.append('keep *_l1extraParticles_MET_*')

	# the gen-level MET
process.Out.outputCommands.append('keep *_genMetTrue_*_*')



process.FEVToutput_step = cms.EndPath(process.Out)







