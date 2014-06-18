# Import configurations
import FWCore.ParameterSet.Config as cms

# Import a helper function for accessing dataset files on EOS
import SLHCUpgradeSimulations.L1TrackTrigger.tools.datasetsHelper as m_datasetsHelper

# set up process
process = cms.Process("Tau")

#
# This configuration runs over a file that contains the L1Tracks
# and the tracker digis.
# It creates L1TkTausFromCalo by clustering together high-pT L1Tracks 
# and matching them to stage-2 L1CaloTaus.
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

# Load Sequences: (Check that the following are all needed)
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff") ###check this for MC!
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/L1HwVal_cff')
# Run the SLHCCaloSequence  to produce the L1Jets
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

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

# Produce calibrated (eT-corrected) L1CaloTaus:
process.L1CaloTauCorrectionsProducer = cms.EDProducer("L1CaloTauCorrectionsProducer", 
     L1TausInputTag = cms.InputTag("SLHCL1ExtraParticles","Taus") 
)

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
      DeltaR_L1TkTauIsolation          = cms.double( 0.40 ),     # Isolation cone size for L1TkTau
      DeltaR_L1TkTau_L1CaloTau         = cms.double( 0.15 ),     # Matching cone for L1TkTau and L1CaloTau
      L1CaloTau_EtMin                  = cms.double( 5.0  ),     # Min eT applied on all L1CaloTaus [GeV]
      RemoveL1TkTauTracksFromIsoCalculation = cms.bool( False ), # Remove tracks used in L1TkTau construction from VtxIso calculation?
)

process.pCorr = cms.Path( process.L1CaloTauCorrectionsProducer )
process.pTaus = cms.Path( process.L1TkTauFromCaloProducer )

# Define the output module
process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "output.root" ), #"L1TkTausFromCalo.root"
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)

# Define the collections to be saved
process.Out.outputCommands.append('keep *_L1CaloTauCorrections*_*_*')
process.Out.outputCommands.append('keep *_L1TkTauFromCalo*_*_*')
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_*_*')
process.Out.outputCommands.append('keep *_gen*_*_*')

process.FEVToutput_step = cms.EndPath(process.Out)




