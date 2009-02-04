import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# tqaf Layer 2 for semi-leptonic event selections
#-------------------------------------------------

## produce ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *

## apply objects count selection for *leptons*
from TopQuarkAnalysis.TopEventProducers.sequences.numLepFilter_ttSemiLeptonic_cff import *

## apply objects count selection for *jets*
from TopQuarkAnalysis.TopEventProducers.sequences.numJetFilter_ttSemiLeptonic_cff import *

## produce kin fit for signal selection
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepKinematicFit_cff import *

## produce mva discriminant for signal selection
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepMVASelection_cff import *

## produce ttSemiEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff import *

## produce event solution (legacy)
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiEvtSolProducer_cfi import *


## make tqaf layer2
tqafLayer2_ttSemiLeptonic = cms.Sequence(makeGenEvt *
                                         makeTtSemiLepKinematicFit +
                                         makeTtSemiLepMVASelDiscriminant +
                                         makeTtSemiLepEvent +
                                         solutions
                                         )

## make tqaf layer2 filtered for full leptonic decays
tqafLayer2_ttSemiLeptonic_fullLepFilter = cms.Sequence(makeGenEvt *
                                                       makeTtSemiLepKinematicFit +
                                                       makeTtSemiLepMVASelDiscriminant +
                                                       makeTtSemiLepEvent_fullLepFilter +
                                                       solutions
                                                       )

## make tqaf layer2 filtered for semi-leptonic decays
tqafLayer2_ttSemiLeptonic_semiLepFilter = cms.Sequence(makeGenEvt *
                                                       makeTtSemiLepKinematicFit +
                                                       makeTtSemiLepMVASelDiscriminant +
                                                       makeTtSemiLepEvent_semiLepFilter +
                                                       solutions
                                                       )

## make tqaf layer2 filtered for full hadronic decays
tqafLayer2_ttSemiLeptonic_fullHadFilter = cms.Sequence(makeGenEvt *
                                                       makeTtSemiLepKinematicFit +
                                                       makeTtSemiLepMVASelDiscriminant +
                                                       makeTtSemiLepEvent_fullHadFilter +
                                                       solutions
                                                       )

