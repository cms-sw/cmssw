# Import configurations
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1simulation")

#
# This configuration runs the SLHCCalo sequence, i.e. it creates 
# L1EG (and L1IsoEG) candidates, L1Taus and L1Jets.
# "l1extra" objects are also created, that correspond to the
# the Run-1 algorithms. 
#
 

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
  '/store/mc/TTI2023Upg14D/SingleElectronFlatPt0p2To50/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/C001841E-A3E5-E311-8C0C-003048678E52.root'
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

# ---  Run the SLHCCaloSequence 

process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_forTTI_cff")

process.L1CalibFilterTowerJetProducer.pTCalibrationThreshold = cms.double(40) # applies calibration only to > 40GeV L1 jets

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

        # run L1Reco to produce the L1 objects corresponding        
	# to the current trigger
#process.load('Configuration.StandardSequences.L1Reco_cff')
process.L1Reco = cms.Path( process.l1extraParticles )




process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "example_L1simulation.root"),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


# generator level information :
process.Out.outputCommands.append('keep *_generator_*_*')
process.Out.outputCommands.append('keep *_genParticles_*_*')

# Collections for L1EG objects :
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_EGamma_*')		# old stage-2 algorithm, 2x2 clustering, inclusive EG
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_IsoEGamma_*')		# old stage 2, Isolated EG
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticlesNewClustering_EGamma_*')	# new stage-2 algo, dynamic clustering, inclusive EG
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticlesNewClustering_IsoEGamma_*')  # new stage-2 algo, dynamic clustering, Isolated EG
process.Out.outputCommands.append('keep *_l1extraParticles_NonIsolated_*')		# Run-1, NonIso EG (different from inclusive !)
process.Out.outputCommands.append('keep *_l1extraParticles_Isolated_*')			# Run-1, Iso EG

# Collections for L1Jets :
process.Out.outputCommands.append('keep *_L1CalibFilterTowerJetProducer_CalibratedTowerJets_*')         # L1Jets
process.Out.outputCommands.append('keep *_L1CalibFilterTowerJetProducer_UncalibratedTowerJets_*')       # L1Jets

# Collections for L1CaloTau objects :
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_Taus_*')		# old stage-2 algo, inclusive Taus
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_IsoTaus_*')	# old stage-2 algo,  isolated Taus

# Collections for (Run-1 like) L1Muons :
process.Out.outputCommands.append('keep *L1Muon*_l1extraParticles__*')		# Run-1 L1Muons

# needed to be able to use the _genParticles_  ...
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)



