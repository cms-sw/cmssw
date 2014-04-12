import FWCore.ParameterSet.Config as cms

#
# produce ttFullHadEvent structure with all necessary ingredients,
# needs ttGenEvent as input
#

## std sequence to produce the ttFullHadEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttFullHadEvtHypotheses_cff import *

## configure ttFullHadEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtFullHadEvtBuilder_cfi import *

### make ttFullHadEvent
#makeTtFullHadEvent = cms.Sequence(makeTtFullHadHypotheses *
                                  #ttFullHadEvent
                                  #)


################################################################################
## helper functions
## (examples of usage can be found in the ttFullHadEvtBuilder_cfg.py)
################################################################################

## add hypotheses to the process
def addTtFullHadHypotheses(process,
                           names
                           ):

    ## edit list of input hypotheses for the TtFullHadEventBuilder
    labels =  getattr(process.ttFullHadEvent, "hypotheses")
    for obj in range(len(names)):
        ## create correct label from HypoClassKey string (stripping the leading "k")
        ## e.g. kKinFit -> ttFullHadHypKinFit
        label = "ttFullHadHyp" + names[obj][1:]
        ## add it to the list
        labels.append(label)
    process.ttFullHadEvent.hypotheses = labels

    ### include hypotheses in the standard sequence
    #sequence = getattr(process, "makeTtFullHadHypotheses")
    #for obj in range(len(names)):
        ### create correct label from HypoClassKey string (stripping the leading "k")
        ### e.g. kKinFit -> makeHypothesis_kinFit
        #if names[obj][1:4] == "MVA":
            #label = "makeHypothesis_" + names[obj][1:4].lower() + names[obj][4:]
        #else:
            #label = "makeHypothesis_" + names[obj][1:2].lower() + names[obj][2:]
        ### add it to the sequence
        #sequence += getattr(process, label)


## remove genMatch hypothesis from the process
def removeTtFullHadHypGenMatch(process):
    #process.makeTtFullHadHypotheses.remove(process.makeHypothesis_genMatch)
    process.ttFullHadEvent.hypotheses.remove("ttFullHadHypGenMatch")


## set a specific attribute for all hypotheses to a given value
## -> this works for "jets", "maxNJets", "jetCorrectionLevel"
def setForAllTtFullHadHypotheses(process, attribute, value):
    modules = ["ttFullHadJetPartonMatch",
               "ttFullHadHypGenMatch",
               "kinFitTtFullHadEventHypothesis",
               "ttFullHadHypKinFit"]
    for obj in range(len(modules)):
        object = getattr(process, modules[obj])
        if hasattr(object, attribute):
            setattr(object, attribute, value)
