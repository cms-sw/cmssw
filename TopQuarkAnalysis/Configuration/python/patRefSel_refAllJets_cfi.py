import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patSequences_cff import *
from TopQuarkAnalysis.Configuration.patRefSel_common_cfi import *

### Muons

### Jets

veryLoosePatJets = selectedPatJets.clone(
  src = 'selectedPatJets'
, cut = '' # veryLooseJetCut
)
loosePatJets = selectedPatJets.clone(
  src = 'veryLoosePatJets'
, cut = '' # looseJetCut
)
tightPatJets = selectedPatJets.clone(
  src = 'loosePatJets'
, cut = '' # tightJetCut
)

step3a = cms.EDFilter(
  "PATCandViewCountFilter"
, src = cms.InputTag( 'selectedPatJets' )
, minNumber = cms.uint32( 6 )
, maxNumber = cms.uint32( 999999 )
)
step3b_1 = step3a.clone( src = 'tightPatJets'    , minNumber = 4 )
step3b_2 = step3a.clone( src = 'loosePatJets'    , minNumber = 5 )
step3b_3 = step3a.clone( src = 'veryLoosePatJets', minNumber = 6 )

step3b = cms.Sequence(step3b_1 *
                      step3b_2 *
                      step3b_3
                      )
