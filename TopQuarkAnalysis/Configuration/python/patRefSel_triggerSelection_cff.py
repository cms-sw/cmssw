import FWCore.ParameterSet.Config as cms

from HLTrigger.special.hltPhysicsDeclared_cfi import *
hltPhysicsDeclared.L1GtReadoutRecordTag = cms.InputTag( '' )

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
triggerResults = triggerResultsFilter.clone( hltResults = cms.InputTag( 'TriggerResults::HLT' )
                                           , l1tResults = cms.InputTag( '' )
                                           , throw      = False
                                           )

triggerSelection = cms.Sequence(
  hltPhysicsDeclared
* triggerResults
)