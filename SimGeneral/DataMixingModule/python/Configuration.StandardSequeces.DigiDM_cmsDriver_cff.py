import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from Configuration.StandardSequences.Digi_cff import *

# Do we want to stop standard digitization right after unsuppressed digis are
# made in order to save cpu?

# If we are going to run this with the DataMixer to follow adding
# detector noise, turn this off for now:

##### #turn off noise in all subdetectors
process.simHcalUnsuppressedDigis.doNoise = False
process.simEcalUnsuppressedDigis.doNoise = False
process.ecal_electronics_sim.doNoise = False
process.es_electronics_sim.doESNoise = False
process.simSiPixelDigis.AddNoise = False
process.simSiStripDigis.Noise = False
process.simMuonCSCDigis.strips.doNoise = False
process.simMuonCSCDigis.wires.doNoise = False
#DTs are strange - no noise flag - only use true hits?
#process.simMuonDTDigis.IdealModel = True
process.simMuonDTDigis.onlyMuHits = True
#
process.simMuonRPCDigis.Noise = False
#####

# "Clean up" digitization to make trigger primitives
# from the new "mixed" calo cells
# and to zero-suppress them for further processing.
#
# Run after the DataMixer only.
#
# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# 
#
# clone these sequences:

DM_EcalTriggerPrimitiveDigis = simEcalTriggerPrimitiveDigis.clone()
DM_EcalDigis = EcalDigis.clone()

# Re-define inputs to point at DataMixer output
DM_EcalTriggerPrimitiveDigis.Label = cms.string('mixData')
DM_EcalTriggerPrimitiveDigis.InstanceEB = cms.string('EBDigiCollectionDM')
DM_EcalTriggerPrimitiveDigis.InstanceEE = cms.string('EEDigiCollectionDM')
#
DM_EcalDigis.digiProducer = cms.string('mixData')
DM_EcalDigis.EBdigiCollection = cms.string('EBDigiCollectionDM')
DM_EcalDigis.EEdigiCollection = cms.string('EEDigiCollectionDM')

ecalDigiSequenceDM = cms.Sequence(DM_EcalTriggerPrimitiveDigis*DM_EcalDigis)

# same for Hcal:

# clone these sequences:

DM_HcalTriggerPrimitiveDigis = simHcalTriggerPrimitiveDigis.clone()
DM_HcalDigis = HcalDigis.clone()

# Re-define inputs to point at DataMixer output
DM_HcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('mixData'),cms.InputTag('mixData'))
DM_HcalDigis.digiLabel = cms.InputTag("mixData")

hcalDigiSequenceDM = cms.Sequence(DM_HcalTriggerPrimitiveDigis+DM_HcalDigis)

doCalDigiDM = cms.Sequence(ecalDigiSequenceDM+hcalDigiSequenceDM)

#
#

PostDM_Digi = cms.Sequence(doCalDigiDM)



