import re

import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.TauTriggerValidator import TauTriggerValidator as _TauTriggerValidator

# GEN-level leg cuts are inferred from the HLT path name: ptMin is the path's
# threshold for that leg minus a margin, so the turn-on stays visible, and a tau
# leg takes its etaMax from the first "_eta<X>p<Y>" token after it. A cut set
# explicitly on a leg wins over the inferred one. e/mu legs must give etaMax.
_ptMargin = {"tau": 10., "mu": 2., "ele": 2.}
_thresholdTokens = {
    "tau": re.compile(r".*PFTauHPS(\d+)"),
    "mu": re.compile(r"(?:Iso)?Mu(\d+)"),
    "ele": re.compile(r"Ele(\d+)"),
}
_etaToken = re.compile(r"eta(\d+)p(\d+)")


def _parsePathCuts(hltPath):
    """-> {objType: {"pt": threshold, "eta": etaMax or None}} from the path name"""
    cuts = {}
    for token in hltPath.split("_"):
        for objType, pattern in _thresholdTokens.items():
            m = pattern.fullmatch(token)
            if m:
                cuts[objType] = {"pt": float(m.group(1)), "eta": None}
        m = _etaToken.fullmatch(token)
        if m and "tau" in cuts and cuts["tau"]["eta"] is None:
            cuts["tau"]["eta"] = float(f"{m.group(1)}.{m.group(2)}")
    return cuts


def _tauTriggerValidator(hltPath, legs):
    cuts = _parsePathCuts(hltPath)
    for leg in legs:
        objType = leg.objType.value()
        if objType not in cuts:
            raise RuntimeError(f"{hltPath}: no '{objType}' threshold found in the path name")
        if not hasattr(leg, "ptMin"):
            leg.ptMin = cms.double(cuts[objType]["pt"] - _ptMargin[objType])
        if not hasattr(leg, "etaMax"):
            if objType != "tau" or cuts["tau"]["eta"] is None:
                raise RuntimeError(f"{hltPath}: give etaMax for the '{objType}' leg explicitly")
            leg.etaMax = cms.double(cuts["tau"]["eta"])
    return _TauTriggerValidator(
        hltProcessName = "HLT",
        hltPath = hltPath,
        label = hltPath,
        outFolder = "HLT/Tau/Validation/" + hltPath,
        ptMax = cms.double(1500.),
        legs = legs,
    )


# DiTau
hltTauTriggerValidationDiTau = _tauTriggerValidator(
    "HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1",
    cms.VPSet(
        cms.PSet(
            objType = cms.string("tau"),
            multiplicity = cms.uint32(2),
            genCollection = cms.InputTag("tauGenJetsSelectorAllHadrons"),
            filterName = cms.string("hltHpsDoublePFTau35MediumDitauWPDeepTau"),
        ),
    ),
)

hltTauTriggerValidationDiTauChargedIso = _tauTriggerValidator(
    "HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1",
    cms.VPSet(
        cms.PSet(
            objType = cms.string("tau"),
            multiplicity = cms.uint32(2),
            genCollection = cms.InputTag("tauGenJetsSelectorAllHadrons"),
            filterName = cms.string("hltHpsDoublePFTau40TrackPt1MediumChargedIsolation"),
        ),
    ),
)

# MuTau
hltTauTriggerValidationMuTau = _tauTriggerValidator(
    "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
    cms.VPSet(
        cms.PSet(
            objType = cms.string("mu"),
            multiplicity = cms.uint32(1),
            etaMax = cms.double(2.1),
            genParticleCollection = cms.InputTag("genParticles"),
            fromTauDecay = cms.bool(True),
            filterName = cms.string("hltL3crIsoL1TkSingleMu22TrkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk"),
        ),
        cms.PSet(
            objType = cms.string("tau"),
            multiplicity = cms.uint32(1),
            genCollection = cms.InputTag("tauGenJetsSelectorAllHadrons"),
            filterName = cms.string("hltHpsPFTau27LooseTauWPDeepTau"),
        ),
    ),
)

# ETau
hltTauTriggerValidationETau = _tauTriggerValidator(
    "HLT_Ele30_WPTight_L1Seeded_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
    cms.VPSet(
        cms.PSet(
            objType = cms.string("ele"),
            multiplicity = cms.uint32(1),
            etaMax = cms.double(2.5),
            genParticleCollection = cms.InputTag("genParticles"),
            fromTauDecay = cms.bool(True),
            filterName = cms.string("hltEle30WPTightGsfTrackIsoL1SeededFilter"),
        ),
        cms.PSet(
            objType = cms.string("tau"),
            multiplicity = cms.uint32(1),
            genCollection = cms.InputTag("tauGenJetsSelectorAllHadrons"),
            filterName = cms.string("hltHpsPFTau30LooseTauWPDeepTau"),
        ),
    ),
)

# SingleTau
hltTauTriggerValidationSingleTau = _tauTriggerValidator(
    "HLT_LooseDeepTauPFTauHPS150_L1NN_eta2p1",
    cms.VPSet(
        cms.PSet(
            objType = cms.string("tau"),
            multiplicity = cms.uint32(1),
            genCollection = cms.InputTag("tauGenJetsSelectorAllHadrons"),
            filterName = cms.string("hltHpsPFTau150LooseTauWPDeepTau"),
        ),
    ),
)

hltTauTriggerValidators = [
    hltTauTriggerValidationDiTau,
    hltTauTriggerValidationDiTauChargedIso,
    hltTauTriggerValidationMuTau,
    hltTauTriggerValidationETau,
    hltTauTriggerValidationSingleTau,
]

hltTauTriggerValidationSequence = cms.Sequence(sum(hltTauTriggerValidators[1:], hltTauTriggerValidators[0]))
