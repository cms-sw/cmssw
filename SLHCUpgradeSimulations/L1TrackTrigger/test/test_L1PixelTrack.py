import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Configuration.StandardSequences.L1HwVal_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_forTTI_cff')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/NoPU/SingleTau1p_E2023TTI_NoPU.root'
    )
)


process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff') ###check this for MC!
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')

#################################################################################################
# L1 Track
#################################################################################################

# for 6.1
# process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
# process.BeamSpotFromSim = cms.EDProducer("BeamSpotFromSimProducer")
# process.TT_step = cms.Path(process.BeamSpotFromSim*process.L1Tracks)

# -- Run the L1Tracking :

process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

process.BeamSpotFromSim =cms.EDProducer("BeamSpotFromSimProducer")

process.load('Configuration.StandardSequences.L1TrackTrigger_cff')

# if one wants to change the extrapolation window :
#process.TTTracksFromPixelDigis.phiWindowSF = cms.untracked.double(2.0)   #  default is 1.0

process.TT_step = cms.Path(process.TrackTriggerTTTracks)
process.TTAssociator_step = cms.Path(process.TrackTriggerAssociatorTracks)

process.load('SLHCUpgradeSimulations.L1TrackTrigger.L1TkJetProducer_cfi')
process.L1TkJetsL1 = process.L1TkJets.clone()

#################################################################################################
# Analyzer
#################################################################################################
#process.L1PixelTrigger = cms.EDAnalyzer('L1PixelTrigger',
#      L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
#      L1TkJetInputTag = cms.InputTag("L1TkJetsL1","Central")
#      #L1TkJetInputTag = cms.InputTag("L1TkJets","Central")
#      #L1TkJetInputTag = cms.InputTag("L1CalibFilterTowerJetProducer","CalibratedTowerJets")
##    tau = cms.InputTag("SLHCL1ExtraParticles","Taus"),
##    egamma = cms.InputTag("SLHCL1ExtraParticlesNewClustering","EGamma")
#)

process.L1PixelTrackFit = cms.EDProducer("L1PixelTrackFit")

process.p = cms.Path(
    process.RawToDigi+
    process.valHcalTriggerPrimitiveDigis+
    process.SLHCCaloTrigger+
    process.L1TkJetsL1+
    process.siPixelRecHits
)

from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
process.siPixelRecHits = siPixelRecHits

process.raw2digi_step = cms.Path(process.RawToDigi)
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# bug fix for missing HCAL TPs in MC RAW
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import HcalTPGCoderULUT
HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)
process.valRctDigis.hcalDigis = cms.VInputTag(cms.InputTag('valHcalTriggerPrimitiveDigis'))
process.L1CaloTowerProducer.HCALDigis = cms.InputTag("valHcalTriggerPrimitiveDigis")


#################################################################################################
# Output file
#################################################################################################
process.TFileService = cms.Service("TFileService", fileName = cms.string('ntuple.root') )


# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.TT_step,process.reconstruction_step,process.p)
#process.schedule = cms.Schedule(process.TT_step,process.L1TkJetsL1,process.p)
#process.schedule = cms.Schedule(process.TT_step,process.L1TkJetsL1)


# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023TTI

#call to customisation function cust_2023TTI imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023TTI(process)

process.p2 = cms.Path(
    process.L1PixelTrackFit
)
