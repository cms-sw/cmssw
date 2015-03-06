# Import configurations
import FWCore.ParameterSet.Config as cms

# set up process
process = cms.Process("L1TkCalo")

#
# This configuration runs over a file that contains the L1Tracks
# and the tracker digis.
# It creates HT and MHT, from the L1Jets and from the HLT HI jets.
#
 

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
'/store/mc/TTI2023Upg14D/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/022FFF01-E4E0-E311-9DAD-002618943919.root'
   )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )


# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    #reportEvery = cms.untracked.int32(500),
    reportEvery = cms.untracked.int32(10),
    limit = cms.untracked.int32(10000000)
)      
       
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# Load geometry
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

# ---- Global Tag :
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')


process.load("Configuration.StandardSequences.Services_cff")
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/L1HwVal_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff") ###check this for MC!
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')


# ---------------------------------------------------------------------------

# ---  Run the SLHCCaloSequence  to produce the L1Jets

process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_forTTI_cff")

#process.L1CalibFilterTowerJetProducer.pTCalibrationThreshold = cms.double(40) # applies calibration only to > 40GeV L1 jets

process.p = cms.Path(
    process.RawToDigi+
    process.SLHCCaloTrigger
    )

# bug fix for missing HCAL TPs in MC RAW
process.p.insert(1, process.valHcalTriggerPrimitiveDigis)
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import HcalTPGCoderULUT
HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)
process.valRctDigis.hcalDigis             = cms.VInputTag(cms.InputTag('valHcalTriggerPrimitiveDigis'))
process.L1CaloTowerProducer.HCALDigis =  cms.InputTag("valHcalTriggerPrimitiveDigis")

#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#
# --- Load the L1TkCaloSequence :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkCaloSequence_cff")

# -- Produce L1TkJets, HT and MHT from the L1Jets :	  
process.L1TkCaloL1Jets = cms.Path( process.L1TkCaloSequence )

# -- Produce the HLT JI Jets and L1TkJets, HT and MHT from  these jets :
process.L1TkCaloHIJets = cms.Path( process.L1TkCaloSequenceHI )

#
# ---------------------------------------------------------------------------




process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_L1TkCalo.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


# generator level information
process.Out.outputCommands.append('keep *_generator_*_*')

#process.Out.outputCommands.append('keep *_l1extraParticles_*_*')   # Run-1 like objects... better not use for HT and MHT.

# intermediate products:
process.Out.outputCommands.append('keep *_iterativeConePu5CaloJets_*_*')	# HLT HI jets
process.Out.outputCommands.append('keep *_L1CalibFilterTowerJetProducer_CalibratedTowerJets_*')	   	# L1Jets
process.Out.outputCommands.append('keep *_L1CalibFilterTowerJetProducer_UncalibratedTowerJets_*')	# L1Jets

# Collections of L1TkJets :
process.Out.outputCommands.append('keep *_L1TkJets_*_*')    	#  L1TkJets from the L1Jets
process.Out.outputCommands.append('keep *_L1TkJetsHI_*_*')    	#  L1TkJets from the HLT Heavy Ion jets

# Collections of HT and MHT variables :

	# -- made from the L1Jets :
process.Out.outputCommands.append('keep *_L1TkHTMissCalo_*_*')	 	# from L1Jets, calo only
process.Out.outputCommands.append('keep *_L1TkHTMissVtx_*_*')   	# from L1Jets, with vertex constraint

	# -- made from the HLT HI jets:
process.Out.outputCommands.append('keep *_L1TkHTMissCaloHI_*_*')   	# from HLT HI jets, calo only
process.Out.outputCommands.append('keep *_L1TkHTMissVtxHI_*_*')   	# from HLT HI jets, with vertex constraint

# keep the L1TkTracks if one needs the tracks associated with the jets 
process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_*')


#process.Out.outputCommands.append('keep *')

process.FEVToutput_step = cms.EndPath(process.Out)





