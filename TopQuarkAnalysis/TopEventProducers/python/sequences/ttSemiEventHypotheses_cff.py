import FWCore.ParameterSet.Config as cms

#
# produce ttSemiLep event hypotheses
#

## geom hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypGeom_cff import *

## wMassmaxSumPt hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypWMassMaxSumPt_cff import *

## maxSumPtWMass hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMaxSumPtWMass_cff import *

## genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypGenMatch_cff import *

## mvaDisc hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMVADisc_cff import *

## kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypKinFit_cff import *

## make all considered event hypotheses
makeTtSemiLepHypotheses  = cms.Sequence(makeHypothesis_geom *
                                        makeHypothesis_wMassMaxSumPt *
                                        makeHypothesis_maxSumPtWMass *
                                        makeHypothesis_genMatch      *
                                        makeHypothesis_mvaDisc       *
                                        makeHypothesis_kinFit
                                        )

