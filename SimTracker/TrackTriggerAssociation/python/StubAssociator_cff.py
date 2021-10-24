#---------------------------------------------------------------------------------------------------------
# This definess StubAssociator which produces StubAssociation which will be used by L1Tigger/trackerDTC and
# trackerTFP Analyzer to associate Stubs with MC truth either by using TTStubAssociationMap or
# TTClusterAssociationMap, where the latter is more useful to debug l1 tracking, the former has been
# implemented to enable use of same association as in standart workflow analyzer if wanted.
#---------------------------------------------------------------------------------------------------------

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.ProducerSetup_cff import TrackTriggerSetup
from SimTracker.TrackTriggerAssociation.StubAssociator_cfi import StubAssociator_params

StubAssociator = cms.EDProducer('tt::StubAssociator', StubAssociator_params)