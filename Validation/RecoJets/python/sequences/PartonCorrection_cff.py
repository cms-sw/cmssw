import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# sequence to for parton closure
#----------------------------------------------

import Validation.RecoJets.producers.PartonCorrection_cfi

## clone modules
b   = Validation.RecoJets.producers.PartonCorrection_cfi.partonCorrection.clone()
c   = Validation.RecoJets.producers.PartonCorrection_cfi.partonCorrection.clone()
uds = Validation.RecoJets.producers.PartonCorrection_cfi.partonCorrection.clone()
qrk = Validation.RecoJets.producers.PartonCorrection_cfi.partonCorrection.clone()
glu = Validation.RecoJets.producers.PartonCorrection_cfi.partonCorrection.clone()

## do proper replacements
b.partons   = [5]
c.partons   = [4]
uds.partons = [1, 2, 3]
qrk.partons = [1, 2, 3, 4, 5]
glu.partons = [21]

## sequences
makeBCorrection    = cms.Sequence(b)
makeCCorrection    = cms.Sequence(c)
makeUdsCorrection  = cms.Sequence(uds)
makeQrkCorrection  = cms.Sequence(qrk)
makeGluCorrection  = cms.Sequence(glu)
makeAllCorrection  = cms.Sequence(b   +
                                  c   +
                                  uds +
                                  qrk +
                                  glu
                                   )


