import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

def _addNoFlow(module):
    _noflowSeen = set()
    for eff in module.efficiency.value():
        tmp = eff.split(" ")
        if "cut" in tmp[0]:
            continue
        ind = -1
        if tmp[ind] == "fake" or tmp[ind] == "simpleratio":
            ind = -2
        if not tmp[ind] in _noflowSeen:
            module.noFlowDists.append(tmp[ind])
        if not tmp[ind-1] in _noflowSeen:
            module.noFlowDists.append(tmp[ind-1])

_defaultSubdirs = ["Tracking/TrackingMCTruth/SimDoublets/general"]

postProcessorSimDoublets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirs),
    efficiency = cms.vstring(
        "efficiency_vs_pT 'SimDoublets efficiency vs p_{T}; TP transverse momentum p_{T} [GeV]; Total fraction of SimDoublets passing all cuts' pass_numVsPt numVsPt",
        "efficiency_vs_eta 'SimDoublets efficiency vs #eta; TP pseudorapidity #eta; Total fraction of SimDoublets passing all cuts' pass_numVsEta numVsEta",
        "efficiencyTP_vs_pT 'TrackingParticle efficiency (2 or more connected SimDoublets passing cuts); TP transverse momentum p_{T} [GeV]; Efficiency for TrackingParticles' pass_numTPVsPt numTPVsPt",
        "efficiencyTP_vs_eta 'TrackingParticle efficiency (2 or more connected SimDoublets passing cuts); TP pseudorapidity #eta; Efficiency for TrackingParticles' pass_numTPVsEta numTPVsEta"
    ),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    makeGlobalEffienciesPlot = cms.untracked.bool(True)
)

_addNoFlow(postProcessorSimDoublets)
