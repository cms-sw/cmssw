import FWCore.ParameterSet.Config as cms

# ECAL validation sequences
#
from Validation.EcalHits.ecalSimHitsValidationSequence_cff import *
from Validation.EcalDigis.ecalDigisValidationSequence_cff import *
from Validation.EcalRecHits.ecalRecHitsValidationSequence_cff import *
from Validation.EcalClusters.ecalClustersValidationSequence_cff import *

ecalSimValid = cms.Sequence(ecalSimHitsValidationSequence+ecalDigisValidationSequence+ecalRecHitsValidationSequence+ecalClustersValidationSequence)

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalMonitorTasks.EcalFEDMonitor_cfi import *
from DQMOffline.Ecal.EcalZmassTask_cfi import *
from DQMOffline.Ecal.EcalPileUpDepMonitor_cfi import *

ecalDQMSequencePhase2 = cms.Sequence(
    ecalMonitorTask +
    ecalFEDMonitor +
    ecalzmasstask +
    ecalPileUpDepMonitor
)

validationECALPhase2 = cms.Sequence(
    ecalSimHitsValidationSequence*
    ecalDigisValidationSequence*
    ecalRecHitsValidationSequencePhase2*
    ecalClustersValidationSequence*
    ecalDQMSequencePhase2
)
