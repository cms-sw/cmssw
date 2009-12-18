import FWCore.ParameterSet.Config as cms

#
# produce ttSemiLepEvent structure with all necessary ingredients,
# needs ttGenEvent as input
#

## std sequence to produce the ttSemiLepEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtHypotheses_cff import *

## configure ttSemiLepEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiLepEvtBuilder_cfi import *

## synchronize maxNJets in all hypotheses
ttSemiLepHypGeom              .maxNJets = ttSemiLepEvent.maxNJets
ttSemiLepHypMaxSumPtWMass     .maxNJets = ttSemiLepEvent.maxNJets
ttSemiLepHypWMassMaxSumPt     .maxNJets = ttSemiLepEvent.maxNJets
ttSemiLepJetPartonMatch       .maxNJets = ttSemiLepEvent.maxNJets
findTtSemiLepJetCombMVA       .maxNJets = ttSemiLepEvent.maxNJets
kinFitTtSemiLepEventHypothesis.maxNJets = ttSemiLepEvent.maxNJets

## make ttSemiLepEvent
makeTtSemiLepEvent = cms.Sequence(makeTtSemiLepHypotheses *
                                  ttSemiLepEvent
                                  )

########################################
## helper functions
########################################

## add hypotheses to the process
def addTtSemiLepHypotheses(process,
                           names
                           ):

    ## edit list of input hypotheses for the TtSemiLepEventBuilder
    labels =  getattr(process.ttSemiLepEvent, "hypotheses")
    for obj in range(len(names)):
        ## create correct label from HypoClassKey string (stripping the leading "k")
        ## e.g. kKinFit -> ttSemiLepHypKinFit
        label = "ttSemiLepHyp" + names[obj][1:]
        ## add it to the list
        labels.append(label)
    process.ttSemiLepEvent.hypotheses = labels

    ## include hypotheses in the standard sequence
    sequence = getattr(process, "makeTtSemiLepHypotheses")
    for obj in range(len(names)):
        ## create correct label from HypoClassKey string (stripping the leading "k")
        ## e.g. kKinFit -> makeHypothesis_kinFit
        if names[obj][1:4] == "MVA":
            label = "makeHypothesis_" + names[obj][1:4].lower() + names[obj][4:]
        else:
            label = "makeHypothesis_" + names[obj][1:2].lower() + names[obj][2:]
        ## add it to the sequence
        sequence += getattr(process, label)

## remove genMatch hypothesis from the process
def removeTtSemiLepHypGenMatch(process):
    process.makeTtSemiLepHypotheses.remove(process.makeHypothesis_genMatch)
    process.ttSemiLepEvent.hypotheses.remove("ttSemiLepHypGenMatch")
