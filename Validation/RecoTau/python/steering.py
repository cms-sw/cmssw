import ROOT
#import FWCore.ParameterSet.Config as cms
import re

algorithms = {
    'hps' : {
        # Before cleaning.
        'producer' : 'hpsPFTauProducer',
        'rawSource' : 'combinatoricRecoTaus',
        'nicename' : 'HPS',
        'discriminators' : [
            "hpsPFTauDiscriminationByDecayModeFinding",
            "hpsPFTauDiscriminationByLooseIsolation",
            "hpsPFTauDiscriminationByMediumIsolation",
            "hpsPFTauDiscriminationByTightIsolation",
            "hpsPFTauDiscriminationAgainstElectron",
            "hpsPFTauDiscriminationAgainstMuon",
        ],
        #'color' : '#c1761a',
        'color' : ROOT.EColor.kGreen + 1,
    },
    'shrinking' : {
        'producer' : 'shrinkingConePFTauProducer',
        'rawSource' : 'shrinkingConePFTauProducer',
        'nicename' : "Shrinking Cone",
        'discriminators' : [
            "shrinkingConePFTauDiscriminationByLeadingTrackFinding",
            "shrinkingConePFTauDiscriminationByLeadingTrackPtCut",
            "shrinkingConePFTauDiscriminationByLeadingPionPtCut",
            "shrinkingConePFTauDiscriminationByIsolation",
            "shrinkingConePFTauDiscriminationByTrackIsolation",
            "shrinkingConePFTauDiscriminationByECALIsolation",
            "shrinkingConePFTauDiscriminationAgainstElectron",
            "shrinkingConePFTauDiscriminationAgainstMuon",
        ],
        #'color' : '#367313',
        'color' : ROOT.EColor.kRed,
    },
    'shrinkingTanc' : {
        'producer' : 'shrinkingConePFTauProducer',
        'rawSource' : 'shrinkingConePFTauProducer',
        'nicename' : 'TaNC',
        'discriminators' : [
            "shrinkingConePFTauDiscriminationByLeadingTrackFinding",
            "shrinkingConePFTauDiscriminationByLeadingPionPtCut",
            "shrinkingConePFTauDiscriminationByTaNCfrOnePercent",
            "shrinkingConePFTauDiscriminationByTaNCfrHalfPercent",
            "shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent",
            "shrinkingConePFTauDiscriminationByTaNCfrTenthPercent",
            "shrinkingConePFTauDiscriminationAgainstElectron",
            "shrinkingConePFTauDiscriminationAgainstMuon",
        ],
        #'color' : '#9339fa',
        'color' : ROOT.EColor.kRed,
    },
    'hpsTanc' : {
        'producer' : 'hpsTancTaus',
        # Before cleaning.
        'rawSource' : 'combinatoricRecoTaus',
        'nicename' : 'Hybrid',
        'discriminators' : [
            "hpsTancTausDiscriminationByLeadingTrackPtCut",
            #'hpsTancTausDiscriminationByDecayModeSelection',
            "hpsTancTausDiscriminationByTancLoose",
            "hpsTancTausDiscriminationByTancMedium",
            "hpsTancTausDiscriminationByTancTight",
            'hpsTancTausDiscriminationAgainstElectron',
            'hpsTancTausDiscriminationAgainstMuon',
        ],
        #'color' : '#3285d8',
        'color' : ROOT.EColor.kBlue
    },
    'hpsTancCuts' : {
        'producer' : 'hpsTancTaus',
        # Before cleaning.
        'rawSource' : 'combinatoricRecoTaus',
        'nicename' : 'Hybrid - cuts',
        'discriminators' : [
            'hpsTancTausDiscriminationByDecayModeSelection',
            'hpsTancTausDiscriminationByLooseIsolation',
            'hpsTancTausDiscriminationByMediumIsolation',
            'hpsTancTausDiscriminationByTightIsolation',
            'hpsTancTausDiscriminationAgainstElectron',
            'hpsTancTausDiscriminationAgainstMuon',
        ],
        #'color' : '#3285d8',
        'color' : ROOT.EColor.kBlue
    },
}

_matcher = re.compile(
    r"\w+Discrimination(Against)*(By)*(?P<nicename>\w+)")
def discriminator_nice_name(discriminator):
    match = _matcher.match(discriminator)
    if not match:
        print "Can't make nice name out of discriminator", discriminator
        return ""
    return match.group('nicename')

comparisons = {
    'hybrid_medium' : {
        'title' : "blah",
        'plots' : [
            ('shrinkingTancGenMatched',
             "shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
            ('hpsTancGenMatched', "hpsTancTausDiscriminationByTancMedium"),
            ('hpsGenMatched', "hpsPFTauDiscriminationByMediumIsolation"),
        ]
    },
    'hybrid_loose' : {
        'title' : "blah",
        'plots' : [
            ('shrinkingTancGenMatched',
             "shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
            ('hpsTancGenMatched', "hpsTancTausDiscriminationByTancLoose"),
            ('hpsGenMatched', "hpsPFTauDiscriminationByLooseIsolation"),
        ]
    },
    'new_old_hps_iso_loose' : {
        'title' : "blah",
        'plots' : [
            ('hpsTancCutsGenMatched', "hpsTancTausDiscriminationByLooseIsolation"),
            ('hpsGenMatched', "hpsPFTauDiscriminationByLooseIsolation"),
        ]
    },
    'new_old_hps_iso_medium' : {
        'title' : "blah",
        'plots' : [
            ('hpsTancCutsGenMatched', "hpsTancTausDiscriminationByMediumIsolation"),
            ('hpsGenMatched', "hpsPFTauDiscriminationByMediumIsolation"),
        ]
    },
}
