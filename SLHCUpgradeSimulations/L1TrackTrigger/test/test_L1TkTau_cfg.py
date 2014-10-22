import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

#
# This reruns the L1Tracking to benefit from the latest updates,
# runs the SLHCCaloSequence and produce the L1EGammaCrystal objects,
# and rune the L1TkEmTau producer.
#


from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *

process.source = cms.Source("PoolSource",
      fileNames = minBiasFiles_p1
	#fileNames = cms.untracked.vstring(
		## VBF H->tautau
      #'/store/mc/TTI2023Upg14D/VBF_HToTauTau_125_14TeV_powheg_pythia6/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v1/00000/00114910-0DE9-E311-B42B-0025905A60F4.root',
     #)
)



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
# --- Produces the L1calo objects : run the SLHCCaloSequence
#
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


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# --- Produce the L1TkEmTau objects

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEmTauSequence_cfi")
process.pL1TkEmTaus = cms.Path( process.TkEmTauSequence )

# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------

# --- Output module :


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)

	# the collection of TkEmTau objects to be used for L1Menu :
process.Out.outputCommands.append('keep *_L1TkEmTauProducer_*_*')


	# gen-level information
#process.Out.outputCommands.append('keep *_generator_*_*')
#process.Out.outputCommands.append('keep *_*gen*_*_*')
#process.Out.outputCommands.append('keep *_*Gen*_*_*')
#process.Out.outputCommands.append('keep *_genParticles_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)







