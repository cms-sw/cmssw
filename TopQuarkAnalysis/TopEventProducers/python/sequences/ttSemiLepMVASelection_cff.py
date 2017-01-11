import FWCore.ParameterSet.Config as cms

#
# make mva event selection for semileptonic events
#

## std sequence to produce an mva discriminant
from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVAComputer_cff import *

## make mva discriminant for event selection
makeTtSemiLepMVASelDiscriminantTask = cms.Task(findTtSemiLepSignalSelMVA)
makeTtSemiLepMVASelDiscriminant = cms.Sequence(makeTtSemiLepMVASelDiscriminantTask)
