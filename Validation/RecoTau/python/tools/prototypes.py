
'''

Common prototypes for Tau Validation modules

Author: Evan K. Friis

'''

import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common

def make_process_adder(process, sequence=None):
    my_sequence = [sequence]
    def process_adder(name, object, verbose=True):
        if verbose:
            print "adding", name
        setattr(process, name, object)
        output = getattr(process, name)
        # Append to sequence if desired
        if my_sequence[0]:
            my_sequence[0] += output
        return output
    return process_adder

tau_plotter = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("replace_me"),
    histograms = common.tau_histograms
)

tau_disc_selector = cms.EDFilter(
    "RecoTauDiscriminatorRefSelector",
    src = cms.InputTag("replace_me"),
    discriminator = cms.InputTag("replace_me"),
    cut = cms.double(0.5),
    filter = cms.bool(False),
)

tau_disc_plotter = cms.EDAnalyzer(
    "RecoTauPlotDiscriminator",
    src = cms.InputTag("replace_me"),
    discriminator = cms.VInputTag(),
    nbins = cms.uint32(300),
    min = cms.double(-1),
    max = cms.double(2),
)

tau_truth_matcher = cms.EDProducer(
    "GenJetMatcher",
    src = cms.InputTag("replace_me"),
    matched = cms.InputTag("selectedTrueHadronicTaus"),
    mcPdgId     = cms.vint32(),                      # n/a
    mcStatus    = cms.vint32(),                      # n/a
    checkCharge = cms.bool(False),
    maxDeltaR   = cms.double(0.15),
    maxDPtRel   = cms.double(3.0),
    # Forbid two RECO objects to match to the same GEN object
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(True),
)

tau_matched_selector = cms.EDFilter(
    "CandViewGenJetMatchRefSelector",
    src = cms.InputTag("replace_me"),
    matching = cms.InputTag("replace_me"),
    filter = cms.bool(False)
)

