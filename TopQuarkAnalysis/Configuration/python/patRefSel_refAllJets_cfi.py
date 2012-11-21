import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patSequences_cff import *
from TopQuarkAnalysis.Configuration.patRefSel_common_cfi import *

from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import intermediatePatMuons
from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import loosePatMuons
from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import tightPatMuons
from TopQuarkAnalysis.Configuration.patRefSel_refMuJets_cfi import goodPatJets

### Jets

step3a = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'goodPatJets' )
, minNumber = cms.uint32( 6 )
, maxNumber = cms.uint32( 999999 )
)
step3b_1 = step3a.clone( src = 'goodPatJets60', minNumber = 4 )
step3b_2 = step3a.clone( src = 'goodPatJets50', minNumber = 5 )
step3b_3 = step3a.clone( src = 'goodPatJets'  , minNumber = 6 )

step3b = cms.Sequence(step3b_1 *
                      step3b_2 *
                      step3b_3
                      )
