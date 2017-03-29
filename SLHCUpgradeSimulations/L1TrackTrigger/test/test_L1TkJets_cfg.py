# Import configurations
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TkJets")

#
# This configuration runs over a file that contains the L1Tracks
# and the tracker digis.
# It creates L1TkJets from the L1Jets and from the HLT HI jets.
#
# so far, the creation of TkJets from L1Jets is commented, as
# we don't have yet the "KIT version" of the L1Jets in 620_SLHC.
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
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')

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

# --- Produce L1TkJets from these L1Jets :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkJetProducer_cfi")
process.L1TkJetsL1 = process.L1TkJets.clone()
process.pL1TkJetsL1 = cms.Path( process.L1TkJetsL1 )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# --- Run the HLT Heavy Ion jets :

#
# The sequences in python/L1TkCaloSequence_cff.py allow to make this shorter.
# Here, things are spelled out, to show the various steps.
# See testL1TkCalo_cfg.py for a minimal, black-box like, configuration file.
# 

# --- Run the calo local reconstruction
process.towerMaker.hbheInput = cms.InputTag("hbheprereco")
process.towerMakerWithHO.hbheInput = cms.InputTag("hbheprereco")
process.reconstruction_step = cms.Path( process.calolocalreco )


# --- Produce the  HLT HeavyIon jets :
process.load("RecoHI.HiJetAlgos.HiRecoJets_TTI_cff")
process.hireco = cms.Path( process.hiRecoJets )

# --- Put them into "L1Jets"
process.L1JetsFromHIHLTJets = cms.EDProducer("L1JetsFromHIHLTJets",
        ETAMIN = cms.double(0),
        ETAMAX = cms.double(3.),
        HIJetsInputTag = cms.InputTag("iterativeConePu5CaloJets")
)
process.pL1Jets = cms.Path( process.L1JetsFromHIHLTJets )

# --- Produce L1TkJets from the HeavyIon jets
process.L1TkJetsHI = process.L1TkJets.clone()
process.L1TkJetsHI.L1CentralJetInputTag = cms.InputTag("L1JetsFromHIHLTJets")
process.pL1TkJetsHI = cms.Path( process.L1TkJetsHI )

#
# ---------------------------------------------------------------------------




process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_TkJets.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)

#process.Out.outputCommands.append( 'keep *_*_*_L1EG' )
process.Out.outputCommands.append('keep *_generator_*_*')

process.Out.outputCommands.append('keep *_L1TkJetsHI_*_*')   	# L1TkJets made from the HLT HeavyIon jets
process.Out.outputCommands.append('keep *_L1TkJetsL1_*_*')      # L1TkJets made from the L1Jets



#process.Out.outputCommands.append('keep *')

process.FEVToutput_step = cms.EndPath(process.Out)



