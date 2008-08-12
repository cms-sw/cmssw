import FWCore.ParameterSet.Config as cms

#
# produce ttSemiLep event hypotheses
#

## geom hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepGeom_cff import *

## wMassmaxSumPt hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepWMassMaxSumPt_cff import *

## maxSumPtWMass hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepMaxSumPtWMass_cff import *

## genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepGenMatch_cff import *

## mvaDisc hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepMVADisc_cff import *

## kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepKinFit_cff import *

## make all considered event hypotheses
makeTtSemiHyps  = cms.Sequence(makeHypothesis_geom *
                               makeHypothesis_wMassMaxSumPt *
                               makeHypothesis_maxSumPtWMass *
                               makeHypothesis_genMatch      *
                               makeHypothesis_mvaDisc       *
                               makeHypothesis_kinFit
                               )

