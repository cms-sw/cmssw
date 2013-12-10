import FWCore.ParameterSet.Config as cms

## produce ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *

## produce kin fit for signal selection
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepKinematicFit_cff import *

## produce mva discriminant for signal selection
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepMVASelection_cff import *

## produce ttSemiEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff import *

### make tqaf layer2
#tqafTtSemiLeptonic = cms.Sequence(makeGenEvt *
                                  #makeTtSemiLepKinematicFit +
                                  #makeTtSemiLepMVASelDiscriminant +
                                  #makeTtSemiLepEvent
                                  #)
