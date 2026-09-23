# Phase-2 IT s-link DQM in one job: RelVal digis -> chip bitstream -> raw -> DQMIO

import os
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

_src = os.path.join(os.environ['CMSSW_BASE'], 'src')
_db = os.path.join(_src, 'EventFilter/Phase2PixelRawToDigi/test/OTandITDTCCablingMap.db')

options = VarParsing.VarParsing('analysis')
options.register('outputDQM', 'slink_all.root',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'DQMIO output file')
options.register('cablingDB', 'sqlite_file:' + _db,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Sqlite cabling-map DB connect string')
options.parseArguments()

files = list(options.inputFiles)
if not files:
    for line in open(os.path.join(_src, 'Validation/SiTrackerPhase2V/test/filelist.txt')):
        line = line.strip()
        if line and not line.startswith('#'):
            files.append(line if '://' in line or line.startswith('file:') else 'root://cms-xrd-global.cern.ch/' + line)

process = cms.Process('ITSLINKDQM', Phase2C17I13M9)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D112Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('DQMServices.Core.DQMStore_cfi')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.MessageLogger.cerr.FwkReport.reportEvery = 200

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))
process.source = cms.Source('PoolSource', fileNames=cms.untracked.vstring(files))

process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = options.cablingDB
process.PoolDBESSource = cms.ESSource('PoolDBESSource', process.CondDB,
    toGet=cms.VPSet(cms.PSet(
        record=cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag=cms.string('DTCCablingMapProducerUserRun'),
    )),
)
process.es_prefer_local_cabling = cms.ESPrefer('PoolDBESSource', '')

process.PixelToBitStreamProducer = cms.EDProducer('PixelToBitStreamProducer',
    src=cms.InputTag('generalTracks'),
    siPixelDigi=cms.InputTag('simSiPixelDigis', 'Pixel'),
)
process.BitStreamToRawProducer = cms.EDProducer('BitStreamToRawProducer',
    Phase2ITChipBitStream=cms.InputTag('PixelToBitStreamProducer'),
)

process.itSlinkDQM = cms.EDProducer('Phase2ITValidateSLink',
    src=cms.InputTag('BitStreamToRawProducer'),
    scaleTBPX=cms.untracked.double(1.07),
    scaleTFPX=cms.untracked.double(1.07),
    scaleTEPX=cms.untracked.double(1.17),
    trigger_rate=cms.untracked.double(750.0e3),
    slink_bandwidth=cms.untracked.double(25.0e9),
    folder=cms.untracked.string('InnerTrackerV/SLink'),
)

process.dqmout = cms.OutputModule('DQMRootOutputModule',
    fileName=cms.untracked.string(options.outputDQM),
)

process.p = cms.Path(process.PixelToBitStreamProducer * process.BitStreamToRawProducer * process.itSlinkDQM)
process.e = cms.EndPath(process.dqmout)
process.schedule = cms.Schedule(process.p, process.e)
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.TryToContinue = cms.untracked.vstring('BitStreamToRawProducer')
