import FWCore.ParameterSet.Config as cms

#----------------------------------------------------------------------------------------
#
#
# addtional reco line for caloTaus in parallel to pfTaus in the tqafLayer1 event content.
# Be aware that this reco line is not in use as long as pfTaus (in the summer08
# production) will be switched off. It needs to be revised afterwards.
#
#
#----------------------------------------------------------------------------------------

## define proper jet correction service here
## from JetMETCorrections.Configuration.GFCorrections_StepEfracParameterization_cff import *

## do proper replacements
## add the label of the JetCorrectionService here
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

L1JetCorrector      = 'none'
L2JetCorrector      = 'none'
L3JetCorrector      = 'GFJetCorrectorIcone5'
L4JetCorrector      = 'none'
L5udsJetCorrector   = 'none'
L5gluonJetCorrector = 'none'
L5cJetCorrector     = 'none'
L5bJetCorrector     = 'none'
L6JetCorrector      = 'none'
L7udsJetCorrector   = 'none'
L7gluonJetCorrector = 'none'
L7cJetCorrector     = 'none'
L7bJetCorrector     = 'none'
