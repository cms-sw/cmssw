import FWCore.ParameterSet.Config as cms

#
# produce ttFullHadEvent structure with all necessary ingredients,
# needs ttGenEvent as input
#

## std sequence to produce the ttFullHadEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttFullHadEvtHypotheses_cff import *

## configure ttFullHadEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtFullHadEvtBuilder_cfi import *

## make ttFullHadEvent
makeTtFullHadEventBase = cms.Sequence(makeTtFullHadHypotheses *
                                      ttFullHadEvent
                                      )
makeTtFullHadEvent = cms.Sequence(makeTtFullHadEventBase)


################################################################################
## helper functions
## (examples of usage can be found in the ttFullHadEvtBuilder_cfg.py)
################################################################################

## add hypotheses to the process
def addTtFullHadHypotheses(process,
                           names
                           ):

    ## edit list of input hypotheses for the TtFullHadEventBuilder
    labels = getattr(process.ttFullHadEvent, "hypotheses")
    for obj in names:
        ## create correct label from HypoClassKey string (stripping the leading "k")
        ## e.g. kKinFit -> ttFullHadHypKinFit
        label = "ttFullHadHyp" + obj[1:]
        ## add it to the list
        labels.append(label)
    process.ttFullHadEvent.hypotheses = labels

    ## include hypotheses in the standard sequence
    sequence = getattr(process, "makeTtFullHadHypotheses")
    for obj in names:
        ## create correct label from HypoClassKey string (stripping the leading "k")
        ## e.g. kKinFit -> makeHypothesis_kinFit
        if obj[1:4] == "MVA":
            label = "makeHypothesis_" + obj[1:4].lower() + obj[4:]
        else:
            label = "makeHypothesis_" + obj[1:2].lower() + obj[2:]
        ## add it to the sequence
        sequence += getattr(process, label)


## clone ttFullHadEvent
def cloneTtFullHadEvent(process
                        ):

    ## search highest already existing clone of ttFullHadEvent
    ## to get the needed index for the new ttFullHadEvent
    i=2
    while ("ttFullHadEvent"+str(i) in process.producerNames()):
        i = i+1
    ## clone the ttFullHadEvent including all hypotheses
    from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
    process.makeTtFullHadEvent += cloneProcessingSnippet(process, process.makeTtFullHadEventBase, str(i))
        

## remove genMatch hypothesis from the process
def removeTtFullHadHypGenMatch(process):
    process.makeTtFullHadHypotheses.remove(process.makeHypothesis_genMatch)
    process.ttFullHadEvent.hypotheses.remove("ttFullHadHypGenMatch")
    ## remove genMatch hypothesis for all clones of the ttFullHadEvent
    i=2
    while ("ttFullHadEvent"+str(i) in process.producerNames()):
        getattr(process, "makeTtFullHadHypotheses"+str(i)).remove(getattr(process, "makeHypothesis_genMatch"+str(i)))
        getattr(process, "ttFullHadEvent"+str(i)).hypotheses.remove(cms.InputTag("ttFullHadHypGenMatch"+str(i)))
        ## remove full ttFullHadEvent sequence if
        ## kGenMatch was the only included hypothesis
        if len(getattr(process, "ttFullHadEvent"+str(i)).hypotheses) == 0:
            process.makeTtFullHadEvent.remove(getattr(process,'makeTtFullHadEventBase'+str(i)))
        i = i+1


## set a specific attribute for all hypotheses to a given value
## -> this works for "jets", "maxNJets", "jetCorrectionLevel"
def setForAllTtFullHadHypotheses(process, attribute, value):
    modules = ["ttFullHadJetPartonMatch",
               "ttFullHadHypGenMatch",
               "kinFitTtFullHadEventHypothesis",
               "ttFullHadHypKinFit"]
    for obj in modules:
        object = getattr(process, obj)
        if hasattr(object, attribute):
            setattr(object, attribute, value)
