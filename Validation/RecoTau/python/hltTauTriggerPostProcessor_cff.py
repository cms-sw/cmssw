import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from Validation.RecoTau.hltTauTriggerValidation_cff import hltTauTriggerValidators as _validators

_efficiencies = (
    ("Eff_L1",         "L1 / Gen",          "{n}_l1",      "{n}"),
    ("Eff_HLT",        "L1+HLT / Gen",      "{n}_hlt",     "{n}"),
    ("Eff_HLTgivenL1", "L1+HLT / L1",       "{n}_hlt",     "{n}_l1"),
    ("Eff_Matched",    "object matched / Gen", "{n}_matched", "{n}"),
)

_legs = ("leg1", "leg2")
_objTypes = ("tau", "mu", "ele")
_breakdown = [("Frac_NoMatch", "no leg matched", "{n}_nomatch", "{n}")]
_breakdown += [
    (f"Frac_Leg{m}{t.capitalize()}Only", f"only leg{m} ({t}) matched", f"{{n}}_leg{m}_{t}_only", "{n}")
    for m in (1, 2)
    for t in _objTypes
]

_vars1D = (("pt", "p_{T}"), ("eta", "#eta"), ("phi", "#phi"))
_vars2D = (("pt_eta", "p_{T}, #eta"), ("pt_phi", "p_{T}, #phi"), ("eta_phi", "#eta, #phi"))

_names1D = [(f"{leg}_{v}", f"{leg} {t}") for leg in _legs for v, t in _vars1D]
_names2D = [(f"{leg}_{v}", f"{leg} {t}") for leg in _legs for v, t in _vars2D]
_names2D.append(("leg1pt_leg2pt", "leg1 p_{T}, leg2 p_{T}"))

# the breakdown is booked with pt and eta only
_namesBreakdown = [(f"{leg}_{v}", f"{leg} {t}") for leg in _legs for v, t in _vars1D[:2]]


def _entries(names, recipes):
    return [
        f"{prefix}_{name} '{title} ({what})' {num.format(n=name)} {den.format(n=name)}"
        for name, title in names
        for prefix, what, num, den in recipes
    ]


# one DQM folder per validated path; a path selecting a single object has no leg2
def _nObjs(v):
    return sum(leg.multiplicity.value() for leg in v.legs)

_twoLegDirs = [v.outFolder.value() for v in _validators if _nObjs(v) >= 2]
_singleLegDirs = [v.outFolder.value() for v in _validators if _nObjs(v) == 1]


hltTauTriggerPostProcessorTwoLeg = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_twoLegDirs),
    efficiency = cms.vstring(
        _entries(_names1D, _efficiencies)
        + _entries(_names2D, _efficiencies)
        + _entries(_namesBreakdown, _breakdown)
    ),
    efficiencyProfile = cms.untracked.vstring(
        [e.replace("Eff_", "EffProfile_", 1) for e in _entries(_names1D, _efficiencies)]
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    outputFileName = cms.untracked.string("")
)

# single-object paths book no leg2, leg1pt_leg2pt, or single-matched-leg breakdown.
_singleNames1D = [("leg1_" + v, "leg1 " + t) for v, t in _vars1D]
_singleNames2D = [("leg1_" + v, "leg1 " + t) for v, t in _vars2D]
_singleNamesBreakdown = [("leg1_" + v, "leg1 " + t) for v, t in _vars1D[:2]]
hltTauTriggerPostProcessorSingleTau = hltTauTriggerPostProcessorTwoLeg.clone(
    subDirs = cms.untracked.vstring(_singleLegDirs),
    efficiency = cms.vstring(
        _entries(_singleNames1D, _efficiencies)
        + _entries(_singleNames2D, _efficiencies)
        + _entries(_singleNamesBreakdown, _breakdown[:1])
    ),
    efficiencyProfile = cms.untracked.vstring(
        [e.replace("Eff_", "EffProfile_", 1) for e in _entries(_singleNames1D, _efficiencies)]
    ),
)

# Retain the entry point used by runValClient_cfg.py.
hltTauTriggerPostProcessor = cms.Sequence(
    hltTauTriggerPostProcessorTwoLeg + hltTauTriggerPostProcessorSingleTau
)
