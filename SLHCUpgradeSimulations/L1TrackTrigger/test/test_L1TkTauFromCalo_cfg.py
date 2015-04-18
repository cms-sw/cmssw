# Import configurations
import FWCore.ParameterSet.Config as cms

# Import a helper function for accessing dataset files on EOS
import SLHCUpgradeSimulations.L1TrackTrigger.tools.datasetsHelper as m_datasetsHelper

# set up process
process = cms.Process("Tau")

#
# This configuration creates L1TkTausFromCalo by clustering together high-pT L1Tracks 
# and matching them to stage-2 L1CaloTaus.
# Th1 L1Tracks are recreated at the beginning, to benefit from the latest
# L1Tracking improvements.
#

from SLHCUpgradeSimulations.L1TrackTrigger.singleTau1pFiles_cfi import *

process.source = cms.Source("PoolSource",
    #fileNames = m_datasetsHelper.GetEosRootFilesForDataset("Neutrino_Pt2to20_gun")  
    fileNames = singleTau1pFiles  # the private files are better as the official sample
				  # for singleTau missed many decay modes

    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10),
    limit = cms.untracked.int32(10000000)
)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# Load geometry
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

# Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')



# ---------------------------------------------------------------------------
#
# --- Recreate the L1Tracks to benefit from the latest updates
#

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TrackingSequence_cfi")
process.pTracking = cms.Path( process.FullTrackingSequence )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# -- the usual L1Calo stuff

# Load Sequences: (Check that the following are all needed)
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff") ###check this for MC!
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/L1HwVal_cff')
# Run the SLHCCaloSequence  to produce the L1Jets
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_forTTI_cff")

# The sequence SLHCCaloTrigger creates "stage-2" L1Taus.
# Two collections are created:
# a) ("SLHCL1ExtraParticles","Taus")
# b) ("SLHCL1ExtraParticles","IsoTaus")
# So far only the ("SLHCL1ExtraParticles","Taus") collection has been used.
# The ("SLHCL1ExtraParticles","IsoTaus") has not been looked yet.
process.p = cms.Path(
    process.RawToDigi+
    process.SLHCCaloTrigger
    )

# bug fix for missing HCAL TPs in MC RAW
process.p.insert(1, process.valHcalTriggerPrimitiveDigis)
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import HcalTPGCoderULUT
HcalTPGCoderULUT.LUTGenerationMode    = cms.bool(True)
process.valRctDigis.hcalDigis         = cms.VInputTag(cms.InputTag('valHcalTriggerPrimitiveDigis'))
process.L1CaloTowerProducer.HCALDigis =  cms.InputTag("valHcalTriggerPrimitiveDigis")

# ---------------------------------------------------------------------------




# ------------------------------------------------------------------------------
#
# --- the caloTau producers :
#	L1CaloTauCorrectionsProducer   does a recalibration of the CaloTaus
#	L1TkTauFromCaloProducer  produces the Tk-matched CaloTaus


process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1CaloTauSequences_cfi")

process.CaloTaus = cms.Path( process.CaloTauSequence )

# ------------------------------------------------------------------------------




# Define the output module
process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "output.root" ), 
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)

# Define the collections to be saved
process.Out.outputCommands.append('keep *_L1CaloTauCorrections*_*_*')
process.Out.outputCommands.append('keep *_L1TkTauFromCalo*_*_*')
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_*_*')
process.Out.outputCommands.append('keep *_gen*_*_*')

process.FEVToutput_step = cms.EndPath(process.Out)




