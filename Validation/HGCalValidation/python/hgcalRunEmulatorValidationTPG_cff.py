import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import *
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
from Validation.HGCalValidation.hgcalValidationTPG_cfi import *

# load DQM
from DQMServices.Core.DQM_cfg import *
from DQMServices.Components.DQMEnvironment_cfi import *
from Configuration.StandardSequences.EDMtoMEAtJobEnd_cff import *
# import DQMStore service
from DQMOffline.Configuration.DQMOffline_cff import *
dqmSaver.workflow = '/validation/' + 'HGCAL' + '/TPG'


hgcalTPGRunEmulatorValidation = cms.Sequence(hgcalTriggerPrimitives*hgcalTrigPrimValidation*dqmSaver)
