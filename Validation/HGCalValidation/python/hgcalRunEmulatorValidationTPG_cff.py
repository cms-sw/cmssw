import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.l1tHGCalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalVFEProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalConcentratorProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer2Producer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalTowerMapProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalTowerProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
from Validation.HGCalValidation.hgcalValidationTPG_cfi import *

# load DQM
from DQMServices.Core.DQM_cfg import *
from DQMServices.Components.DQMEnvironment_cfi import *
from Configuration.StandardSequences.EDMtoMEAtJobEnd_cff import *

onlineSaver = cms.EDAnalyzer("DQMFileSaverOnline",
    producer = cms.untracked.string('DQM'),
    path = cms.untracked.string('./'),
    tag = cms.untracked.string('validation_HGCAL_TPG')
)

hgcalTPGRunEmulatorValidation = cms.Sequence(L1THGCalTriggerPrimitives*L1THGCalTrigPrimValidation*onlineSaver)
