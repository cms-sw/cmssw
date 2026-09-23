# Harvest the s-link DQMIO: occupancy summaries plus rendered pdf/png via TrackerPhase2PlotUtil

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.register('inputDQM', 'file:slink_all.root',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'DQMIO file produced by ValidateSLink_cfg.py')
options.register('plotDir', 'pdf_slink_all',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Directory for the rendered pdf/png files')
options.parseArguments()

process = cms.Process('HARVEST')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('DQMServices.Core.DQMStore_cfi')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source('DQMRootSource', fileNames=cms.untracked.vstring(options.inputDQM))

process.slinkOccupancyHarvester = cms.EDProducer('SlinkOccupancyHarvester',
    TopFolder        = cms.string('Phase2IT/RawDataSLink'),
    OccupancyMapName = cms.string('slinkOccupancyMap'),
    savePlots     = cms.untracked.bool(True),
    plotDir       = cms.untracked.string(options.plotDir),
    plotFormats   = cms.untracked.vstring('pdf', 'png'),
    plotZMax      = cms.untracked.double(1.6),
    occupancyAvg = cms.PSet(
        switch = cms.bool(True),
        name   = cms.string('slinkOccupancyAvg'),
        title  = cms.string('Event-Averaged SLink Occupancy;<Occupancy>;SLink entries'),
        NxBins = cms.int32(33),
        xmin   = cms.double(0.),
        xmax   = cms.double(1.6),
    ),
)

process.harvest_step = cms.Path(process.slinkOccupancyHarvester)
process.dqmsave_step = cms.Path(process.DQMSaver)
process.schedule = cms.Schedule(process.harvest_step, process.dqmsave_step)
process.options.numberOfThreads = cms.untracked.uint32(1)
