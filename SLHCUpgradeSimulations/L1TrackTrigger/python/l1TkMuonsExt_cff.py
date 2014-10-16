import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.L1TrackTrigger.l1TkMuonsExt_cfi import l1TkMuonsExt
l1TkMuonsExtCSC = l1TkMuonsExt.clone(
    L1MuonsInputTag = cms.InputTag("l1extraMuExtended", "csc"),
    ETAMIN = cms.double(1.1)
    )

l1TkMuonsExtNoZCor = l1TkMuonsExt.clone( correctGMTPropForTkZ = cms.bool(False) )
l1TkMuonsExtCSCNoZCor = l1TkMuonsExtCSC.clone( correctGMTPropForTkZ = cms.bool(False) )

l1TkMuonsExtSequence = cms.Sequence (l1TkMuonsExt * l1TkMuonsExtCSC
                                     * l1TkMuonsExtNoZCor * l1TkMuonsExtCSCNoZCor)

