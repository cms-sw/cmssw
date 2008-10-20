import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# sequence to for calibration comparisons
#----------------------------------------------

import Validation.RecoJets.producers.CompareCalibs_cfi

## clone modules
gfVsFact = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()
factVsGf = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()

## do proper replacements
gfVsFact.recs = 'gfCorICone5'
gfVsFact.refs = 'L2L3CorJetIcone5'
factVsGf.recs = 'L2L3CorJetIcone5'
factVsGf.refs = 'gfCorICone5'

## sequences
makeGfVsFactComparison = cms.Sequence(gfVsFact)
makeFactVsGfComparison = cms.Sequence(factVsGf)
makeAllComparison      = cms.Sequence(gfVsFact +
                                      factVsGf
                                      )

