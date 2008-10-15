import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# sequence to for calibration comparisons
#----------------------------------------------

import Validation.RecoJets.producers.CompareCalibs_cfi

## clone modules
gfVsFact = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()
factVsGf = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()

## do proper replacements
gfVsFact.recs = 'uhhCaliIterativeCone5'
gfVsFact.refs = 'L3JetCorJetIcone5'
factVsGf.recs = 'L3JetCorJetIcone5'
factVsGf.refs = 'uhhCaliIterativeCone5'

## sequences
makeGfVsFactComparison = cms.Sequence(gfVsFact)
makeFactVsGfComparison = cms.Sequence(factVsGf)


