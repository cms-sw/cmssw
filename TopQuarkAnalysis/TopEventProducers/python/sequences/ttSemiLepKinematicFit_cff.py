import FWCore.ParameterSet.Config as cms

#
# make kinematic fit for selection of semileptonic events
#

## std sequence to perform kinematic fit
import TopQuarkAnalysis.TopKinFitter.TtSemiLepKinFitProducer_Muons_cfi
kinFitTtSemiLepEventSelection = TopQuarkAnalysis.TopKinFitter.TtSemiLepKinFitProducer_Muons_cfi.kinFitTtSemiLepEvent.clone()

## make kin fit for event selection
makeTtSemiLepKinematicFit = cms.Sequence(kinFitTtSemiLepEventSelection)
