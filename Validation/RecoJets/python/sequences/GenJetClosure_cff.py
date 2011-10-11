import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# sequence to for parton closure
#----------------------------------------------

import Validation.RecoJets.producers.GenJetClosure_cfi

## clone modules
noCalib = Validation.RecoJets.producers.GenJetClosure_cfi.genJetClosure.clone()
gfCalib = Validation.RecoJets.producers.GenJetClosure_cfi.genJetClosure.clone()
l2Calib = Validation.RecoJets.producers.GenJetClosure_cfi.genJetClosure.clone()
l3Calib = Validation.RecoJets.producers.GenJetClosure_cfi.genJetClosure.clone()

## do proper replacements
noCalib.recs = 'iterativeCone5CaloJets'
gfCalib.recs = 'gfCorJetIcone5'
l2Calib.recs = 'L2L3CorJetIcone5'
l3Calib.recs = 'L2L3CorJetIcone5'

## sequences
makeNoCalibClosure  = cms.Sequence(noCalib)
makeGfCalibClosure  = cms.Sequence(gfCalib)
makeL2CalibClosure  = cms.Sequence(l2Calib)
makeL3CalibClosure  = cms.Sequence(l3Calib)
makeAllCalibClosure = cms.Sequence(noCalib +
                                   gfCalib +
                                   l2Calib +
                                   l3Calib
                                   )
