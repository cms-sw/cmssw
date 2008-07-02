import FWCore.ParameterSet.Config as cms

# PATLayer1 MET input
allLayer1METs.metSource = 'allLayer0METs'
allLayer1METs.addGenMET = True
allLayer1METs.genMETSource = 'genMet'
allLayer1METs.addTrigMatch = False
# replace allLayer1METs.trigPrimMatch      = { metTrigMatchHLT1MET65 }
allLayer1METs.addResolutions = True
allLayer1METs.useNNResolutions = False
allLayer1METs.metResoFile = 'PhysicsTools/PatUtils/data/Resolutions_met.root'
# PATLayer1 MET selection
selectedLayer1METs.src = 'allLayer1METs'
selectedLayer1METs.cut = 'et > 0.'

