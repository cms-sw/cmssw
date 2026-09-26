# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# DQM harvesting for the Branch performance-plot validators: turns the booked
# numerator/denominator histograms into reproduction-efficiency plots vs
# eta/pt/energy, in the same fashion as the standard HGCAL/tracking post-processors
# (DQMGenericClient). Profiles (purity/completeness/response) are produced directly
# by the analyzer and need no harvesting.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

_branchEfficiency = cms.vstring(
    "efficiency_eta 'Branch reproduction efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
    "efficiency_pt 'Branch reproduction efficiency vs p_{T};p_{T} [GeV];efficiency' effnum_pt denom_pt",
    "efficiency_energy 'Branch reproduction efficiency vs E;E [GeV];efficiency' effnum_energy denom_energy",
    # "Other way around": fraction of objects whose best hit-matched Branch is the
    # natural (trackId-seeded) one, from the validator's selfmatch_* numerators.
    "selfmatchrate_eta 'Best Branch is the natural one vs #eta;#eta;self-match rate' selfmatch_eta denom_eta",
    "selfmatchrate_pt 'Best Branch is the natural one vs p_{T};p_{T} [GeV];self-match rate' selfmatch_pt denom_pt",
)

branchHGCalPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        "HGCAL/BranchValidator/CaloParticle",
        "HGCAL/BranchValidator/SimCluster",
    ),
    efficiency=_branchEfficiency,
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

# Tracking: "Branch reproduces the TrackingParticle track->truth assignment"
# efficiency vs eta/pt (energy is meaningless for the tracker).
_branchTrackingEfficiency = cms.vstring(
    "efficiency_eta 'Branch reproduction efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
    "efficiency_pt 'Branch reproduction efficiency vs p_{T};p_{T} [GeV];efficiency' effnum_pt denom_pt",
)

branchTrackingPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        "Tracking/BranchValidator/TrackingParticle",
    ),
    efficiency=_branchTrackingEfficiency,
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

truthGraphDQMHarvesting = cms.Sequence(branchHGCalPostProcessor + branchTrackingPostProcessor)

