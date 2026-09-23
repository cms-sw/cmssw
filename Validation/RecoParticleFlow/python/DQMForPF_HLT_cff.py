import FWCore.ParameterSet.Config as cms

# HLT flavour of the ParticleFlow DQM.
#
# The offline sequence (DQMForPF_MiniAOD_cff) reads MiniAOD: pat::Jet and
# pat::PackedCandidate.  HLT produces neither, so only the two candidate-level
# analyzers are provided here, pointed at the HLT collections:
#
#   hltParticleFlowTmp        the full HLT PF candidate collection; it is what
#                             hltAK4PFPuppiJets is clustered from, so it is the
#                             analogue of offline particleFlow/packedPFCandidates
#   hltOfflinePrimaryVertices the vertices used by the HLT PF chain
#
# The jet-level part of the offline sequence is deliberately not reproduced:
# Validation/RecoJets already provides hltJetValidation for
# hltAK4PFPuppiJets / hltAK4PFJets / hltAK4PFCHSJets.
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetDQMPostProcessor

# The offset plots are booked one per (candidate type, mu) and (type, npv), and
# the analyzer returns without filling if the event's mu or npv has no
# corresponding plot.  The defaults stop at 100, so at 200 PU EVERY event falls
# outside and all offset plots come out empty -- measured: mu = 200 against an
# axis ending at 100.  Rebuild the VPSet over Phase-2 ranges.
import Validation.RecoParticleFlow.defaults_cfi as default
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import plotPSet

MU_HIGH_PHASE2 = 260
NPV_HIGH_PHASE2 = 160

def createOffsetVPSetPhase2():
    plots = []
    for pftype in default.candidateType:
        for mu in range(default.muLowOffset, MU_HIGH_PHASE2):
            name = default.offset_name("mu", mu, pftype)
            plots += [plotPSet(name, name + ";#eta;<Offset Energy_{T}> [GeV]",
                               "muPlots/mu{0}".format(mu),
                               0, 0, 0, default.eBinsOffset, default.eLowOffset,
                               default.eHighOffset, default.etaBinsOffset)]
        for npv in range(default.npvLowOffset, NPV_HIGH_PHASE2):
            name = default.offset_name("npv", npv, pftype)
            plots += [plotPSet(name, name + ";#eta;<Offset Energy_{T}> [GeV]",
                               "npvPlots/npv{0}".format(npv),
                               0, 0, 0, default.eBinsOffset, default.eLowOffset,
                               default.eHighOffset, default.etaBinsOffset)]
    return plots

# useAOD: reco::PFCandidate input, so PV attachment is decided by matching the
# candidate trackRef against the tracks fitted to a good primary vertex, rather
# than reading pat::PackedCandidate::fromPV().
offsetAnalyzerDQMHLT = offsetAnalyzerDQM.clone(
    dqmDir = "HLT/ParticleFlow/Offset/",
    pfTag  = "hltParticleFlowTmp",
    pvTag  = "hltOfflinePrimaryVertices",
    muTag  = "addPileupInfo",
    muHigh = MU_HIGH_PHASE2,
    npvHigh = NPV_HIGH_PHASE2,
    offsetPlots = createOffsetVPSetPhase2(),
)


# ---------------------------------------------------------------------------
# DQMOffline/ParticleFlow PFAnalyzer, HLT instance.
#
# This one needs no code change at all: it is dual-mode via isMiniAOD, and with
# isMiniAOD=False it reads reco::PFCandidate / reco::PFJet / reco::Vertex
# directly, which is exactly what HLT produces.
#
# The cut list uses the fine |eta| binning around the HGCAL edge (2.6, 2.7, 2.8,
# 2.9, 3.0) that the package's own config carries commented out -- that is the
# region where JME reported the TICLv5 jet structure, so the PF candidate
# spectra are binned to resolve it.
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfAnalyzerDQM

pfAnalyzerDQMHLT = pfAnalyzerDQM.clone(
    isHLT               = True,
    isMiniAOD           = False,
    pfCandidates        = "hltParticleFlowTmp",
    pfJetCollection     = "hltAK4PFPuppiJets",
    PVCollection        = "hltOfflinePrimaryVertices",
    puppiWeight         = "hltPFPuppi",
    TriggerResultsLabel = cms.InputTag("TriggerResults", "", "reHLT"),
    # passesTriggerSelection accepts everything only if the list contains an
    # EMPTY STRING; an empty list makes its inner loop never run, so every
    # event is rejected and nothing fills.
    TriggerNames        = [""],
    eventSelection      = "nocut",     # W' is not a dijet sample
    pfAnalysis = cms.PSet(
        NPVBins = cms.vdouble(0, 100, 250),
        observables = cms.vstring(
            'pt;p_{T,PFC};50.;0.;350.',
            'eta;#eta;50;-5;5',
            'phi;#phi;50;-3.14;3.14',
            'energy;E;50;0;300',
        ),
        eventObservables = cms.vstring(),
        pfInJetObservables = cms.vstring(),
        binList2D = cms.vstring(
            '[eta;30;-5;5][phi;30;-3.14;3.14]',
            '[eta;30;-5;5][logPt;50;-1.5;4.]',
            '[eta;30;-5;5][pt;100;0;10.]',
            '[eta;30;-5;5][energy;50;0;50.]',
        ),
        # inclusive, then resolved in |eta| across the HGCAL edge
        cutList = cms.vstring(
            '[pt;1;0;10000]',
            '[pt;1;0;10000][abseta;0;1.47;4.;6.]',
            '[pt;0;1;2;4;6;10;20;50;100][abseta;0;1.47;4.;6.]',
        ),
        jetCutList = cms.vstring('[pt;20;10000]'),
    ),
)

offsetDQMPostProcessorHLT = offsetDQMPostProcessor.clone(
    offsetDir = "HLT/ParticleFlow/Offset/"
)

DQMHLTPF = cms.Sequence(
    pfAnalyzerDQMHLT +
    offsetAnalyzerDQMHLT
)

DQMHarvestHLTPF = cms.Sequence(
    offsetDQMPostProcessorHLT
)
