#! /usr/bin/env cmsRun
# cmsRun listIds_PhaseII.py fromDB=False

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

###################################################################
# Set default phase-2 settings
###################################################################
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

process = cms.Process("MaterialAnalyser",_PH2_ERA)

options = VarParsing('analysis')

options.register('fromDB',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 'Read Geometry from DB?',
)

options.parseArguments()

if options.fromDB :
   process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
   from Configuration.AlCa.GlobalTag import GlobalTag
   process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')
else:
   process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')
   process.trackerGeometry.applyAlignment = False # needed to avoid to pass the Global Position Record

process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
        firstRun = cms.untracked.uint32(1)
)

process.listIds = cms.EDAnalyzer("ListIds",
                                materials = cms.untracked.vstring("materials:Silicon", "tracker:tkLayout_SenSi"),
                                printMaterial = cms.untracked.bool(True)
                                )
process.path = cms.Path(process.listIds)


