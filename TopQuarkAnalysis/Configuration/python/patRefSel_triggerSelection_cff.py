import FWCore.ParameterSet.Config as cms

l1Tag  = cms.InputTag( '' ) # skip L1 results, since conflicts with the GlobalTag can occur
hltTag = cms.InputTag( 'TriggerResults::HLT' )

from HLTrigger.special.hltPhysicsDeclared_cfi import *
hltPhysicsDeclared.L1GtReadoutRecordTag = l1Tag

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
triggerResults = triggerResultsFilter.clone( hltResults = hltTag
                                           , l1tResults = l1Tag
                                           , throw      = False
                                           )

triggerSelection = cms.Sequence(
  hltPhysicsDeclared
* triggerResults
)