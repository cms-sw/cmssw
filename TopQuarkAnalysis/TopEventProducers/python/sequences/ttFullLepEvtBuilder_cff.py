import FWCore.ParameterSet.Config as cms


################################################################################
# produce ttFullLepEvent structure with all necessary ingredients
################################################################################

## std sequence to produce the ttFullLepEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttFullLepEvtHypotheses_cff import *

## configure ttFullLepEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtFullLepEvtBuilder_cfi import *

### make ttFullLepEvent
#makeTtFullLepEvent = cms.Sequence(makeTtFullLepHypotheses *
                                  #ttFullLepEvent
                                  #)


################################################################################
## helper functions
## (examples of usage can be found in the ttFullLepEvtBuilder_cfg.py)
################################################################################

## remove genMatch hypothesis from the process
def removeTtFullLepHypGenMatch(process):
    #process.makeTtFullLepHypotheses.remove(process.makeHypothesis_genMatch)
    process.ttFullLepEvent.hypotheses.remove("ttFullLepHypGenMatch")
    process.ttFullLepEvent.genEvent = ''


## set a specific attribute for all hypotheses to a given value
## -> this works for "jets", "leps", "mets", "maxNJets"
def setForAllTtFullLepHypotheses(process, attribute, value):
    modules = ["ttFullLepJetPartonMatch",
               "ttFullLepHypGenMatch",
               "ttFullLepHypKinSolution",
	       "kinSolutionTtFullLepEventHypothesis"]
    for obj in range(len(modules)):
        object = getattr(process, modules[obj])
        if hasattr(object, attribute):
            setattr(object, attribute, value)



