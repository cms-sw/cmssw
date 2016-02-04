import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# sequence to for calibration comparisons
#----------------------------------------------

import Validation.RecoJets.producers.CompareCalibs_cfi

#
# compare official global fit versus factorized approach
#

## clone modules
gfVsFact = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()
factVsGf = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()

## do proper replacements
gfVsFact.recs = 'gfCorJetIcone5'
gfVsFact.refs = 'L2L3CorJetIcone5'
factVsGf.recs = 'L2L3CorJetIcone5'
factVsGf.refs = 'gfCorJetIcone5'

## sequences
makeGfVsFactComparison = cms.Sequence(gfVsFact)
makeFactVsGfComparison = cms.Sequence(factVsGf)
makeAllFactComparison  = cms.Sequence(gfVsFact +
                                      factVsGf
                                      )


#
# compare official global fit versus private global fit
#

## clone modules
gfVsPriv = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()
privVsGf = Validation.RecoJets.producers.CompareCalibs_cfi.compareCalibs.clone()

## do proper replacements
gfVsPriv.recs = 'gfCorJetIcone5'
gfVsPriv.refs = 'gfCorIcone5'     # rename jet collections in Calibration/CalibMaker/python/sequences/calibJets_cff.py
privVsGf.recs = 'gfCorIcone5'     # accordingly from 'gfCorJetIcone5' to 'gfCorIcone5' to make this work properly
privVsGf.refs = 'gfCorJetIcone5'

## sequences
makeGfVsPrivComparison = cms.Sequence(gfVsPriv)
makePrivVsGfComparison = cms.Sequence(privVsGf)
makeAllPrivComparison  = cms.Sequence(gfVsPriv +
                                      privVsGf
                                      )
