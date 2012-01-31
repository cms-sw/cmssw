import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.tools.prototypes as proto

def basejet_matching(process, algorithms, genjets, sequence):
    '''
    Match the underlying jet collections to truth to support embedding.
    '''
    builder = proto.make_process_adder(process, sequence)
    for algorithm in algorithms.keys():
        # Match the base jet refs of the algorithm to support embedding if
        # desired.
        jetrefs = getattr(process, algorithms[algorithm]['rawSource']).jetSrc
        jetmatcher = proto.tau_truth_matcher.clone(
            src = jetrefs,
            matched = cms.InputTag(genjets),
        )
        jetmatcher_name = algorithm + "BaseJetsMatcher"
        algorithms[algorithm]['jet_matcher'] = jetmatcher_name
        builder(jetmatcher_name, jetmatcher)


def make_matching(process, algorithms, genjets, sequence):
    '''
    Add a matching (named [algorithm]+"Matcher") between each algorithm (the keys
    in [algorithms]) and the [genjets] collection.  A ref selection of each
    collection in [algorithms] containing only those with valid matches is
    produced in [algorithm]+"GenMatched"

    The function modifies the input algorithms dictionary and adds they keys
    ['matcher'], ['matched_collection'], which point to the new modules.

    '''
    builder = proto.make_process_adder(process, sequence)
    for algorithm in algorithms.keys():
        # Match the actual taus to truth.
        matcher = proto.tau_truth_matcher.clone(
            src = cms.InputTag(algorithms[algorithm]['producer']),
            matched = cms.InputTag(genjets)
        )
        matcher_name = algorithm+"Matcher"
        algorithms[algorithm]['matcher'] = matcher_name
        builder(matcher_name, matcher)
        # Make match selector
        match_selector = proto.tau_matched_selector.clone(
            src = cms.InputTag(algorithms[algorithm]['producer']),
            matching = cms.InputTag(matcher_name)
        )
        selector_name = algorithm + "GenMatched"
        algorithms[algorithm]['matched_collection'] = selector_name
        builder(selector_name, match_selector)
