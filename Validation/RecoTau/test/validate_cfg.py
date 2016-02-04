#!/usr/bin/env cmsRun

'''

validate_cfg

Build basic histograms for validating the RecoTauTag package.

Author: Evan K. Friis


'''

import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.tools.validator import validate
from Validation.RecoTau.tools.matcher import make_matching, basejet_matching
import Validation.RecoTau.steering as steering
import Validation.RecoTau.tools.prototypes as proto
import os
import sys

from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')

options.register(
    'mc', 1,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Specify whether to use generator information"
)

options.register(
    'mcJets', "trueHadronicTaus",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "If [mc] is set true, "
)

options.register(
    'embedmc', 1,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "If true, truth information will be embedded into the reco taus."
    " Requires both [rereco] and [mc] to be enabled."
)

options.register(
    'db', '',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Get HPSTanc training from given sqlite file.  Only valid if [rereco]=1"
)

options.register(
    'transform', '',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Use a custom TaNC transformation.  Only valid if [rereco]=1"
)

options.register(
    'rereco', 0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "If rereco=1, the tau sequences will be rerun."
    "  Otherwise, it assumed the collections are already in the file."
)

options.register(
    'skip', 0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Skip events?"
)

options.register(
    'gt', '',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Use given global tag. If not specified the default MC conditions are taken"
)

options.parseArguments()

process = cms.Process("validate")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    skipEvents = cms.untracked.uint32(options.skip),
)

# Setup output file for the plots
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
)

# We keep track of information aobut each algorithm in the dictionary.  The
# various process modifier functions add information as they mutate the process.
# Load algorithm defintions
algorithms = steering.algorithms

process.dummy = cms.EDProducer("DummyModule")
process.truth = cms.Sequence(process.dummy)
if options.mc:
    print "Using MC truth information.  Truth objects:", options.mcJets
    # For the PDT record.
    process.load("Configuration.StandardSequences.Services_cff")
    process.load("RecoTauTag.TauTagTools.TauTruthProduction_cfi")
    process.truth = process.tauTruthSequence
    # Add plots of the MC truth
    process.plotTruth = proto.tau_plotter.clone(
        src = cms.InputTag(options.mcJets),
        histograms = proto.common.kin_plots
    )
    process.truth += process.plotTruth

process.rereco = cms.Sequence(process.dummy)
# Check if we need to load any of that normal garbage
if options.rereco:
    print "Rerunning tau reconstruction sequences..."
    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load("Configuration.StandardSequences.Services_cff")
    process.load("Configuration.StandardSequences.Geometry_cff")
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.load("Configuration.StandardSequences.MagneticField_cff")
    if options.gt:
        process.GlobalTag.globaltag = "%s::All" % options.gt
        print "Using global tag:", process.GlobalTag.globaltag
    else:
        from Configuration.PyReleaseValidation.autoCond import autoCond
        process.GlobalTag.globaltag = autoCond['mc']
        print "Autoloaded global tag:", autoCond['mc']
    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    process.load("RecoTauTag.Configuration.RecoTauTag_cff")
    process.rereco = cms.Sequence(process.PFTau)
    process.shrinkingConePFTauProducer.outputSelection = cms.string(
        'pt > 20 & abs(eta) < 2.5')
    process.hpsTancTaus.outputSelection = cms.string(
        'pt > 20 & abs(eta) < 2.5')
    process.hpsPFTauProducer.outputSelection = cms.string(
        'pt > 20 & abs(eta) < 2.5')
    if options.db:
        # Load custom database
        print "Loading custom MVA conditions"
        process.load("RecoTauTag.TauTagTools.TancConditions_cff")
        process.TauTagMVAComputerRecord.connect = cms.string(
            'sqlite_fip:%s' % options.db)
        process.TauTagMVAComputerRecord.toGet[0].tag = "Tanc"
        process.TauTagMVAComputerRecord.appendToDataLabel = cms.string(
            'tau_validation')
        # Don't conflict with GlobalTag (which provided shrinking cone tanc)
        process.combinatoricRecoTausDiscriminationByTanc.dbLabel = \
                "tau_validation"
        process.hpsTancTausDiscriminationByTancRaw.dbLabel = \
                "tau_validation"
    if options.transform:
        print "Loading custom transform"
        transform_dir = os.path.dirname(options.transform)
        print transform_dir
        sys.path.append(transform_dir)
        import transforms as custom_transform
        # Set the TaNC transformed to use the input transform
        process.hpsTancTausDiscriminationByTanc.transforms = \
                custom_transform.transforms
        process.combinatoricRecoTausTancTransform.transforms = \
                custom_transform.transforms


process.matching = cms.Sequence(process.dummy)
if options.mc:
    print "Building matching"
    # Build a matching for each of our algorithms
    make_matching(process, algorithms, options.mcJets, process.matching)
#print process.matching

already_embedded = set()
# Check if we want to embed the truth into our taus.
process.jetmatching = cms.Sequence(process.dummy)
if options.embedmc and options.mc and options.rereco:
    # Build basejet (i.e. the seed PFJet) matching sequence
    basejet_matching(process, algorithms, options.mcJets, process.jetmatching)
    for algorithm in algorithms.keys():
        raw_source = algorithms[algorithm]['rawSource']
        print "Embedding MC truth information into the " + algorithm + ""
        "collection.  The base tau collection modified is" + raw_source
        # Don't embed if we already have done it.
        if raw_source in already_embedded:
            "Truth content already embedded - skipping!"
            continue
        already_embedded.add(raw_source)
        #Get the EDProducer that does the clean
        algorithm_module = getattr(process, algorithms[algorithm]['rawSource'])
        # Build the RecoTauModifierPlugin PSet
        algorithm_embedder = cms.PSet(
            # Get the input matching.
            jetTruthMatch = cms.InputTag(algorithms[algorithm]['jet_matcher']),
            name = cms.string('embed'),
            plugin = cms.string('RecoTauTruthEmbedder'),
        )
        algorithm_module.modifiers.append(algorithm_embedder)

process.tauvalidator = cms.Sequence(process.dummy)
# Building validation sequences
for algorithm in algorithms.keys():
    print "Building validation sequence for:", algorithm
    algorithm_info = algorithms[algorithm]
    validate(process, algorithm, algorithm_info, process.tauvalidator,
             algorithm_info['discriminators'])

# If we have MC truth information, also produce a validation sequence chain
# where the intitial collection has been matched to GenLevel
if options.mc:
    for algorithm in algorithms.keys():
        print "Building gen-matched validation sequence for:", algorithm
        algo_info = algorithms[algorithm]
        # The intital matching has been done in make_matching.  We just need to
        # update that as the base source of the taus.
        algo_info['producer'] = algo_info['matched_collection']
        validate(process, algorithm, algo_info, process.tauvalidator,
                 algo_info['discriminators'], suffix="GenMatched")


# Run all our stuff.
process.main = cms.Path(
    process.truth
    +process.jetmatching
    +process.rereco
    +process.matching
    +process.tauvalidator
)
