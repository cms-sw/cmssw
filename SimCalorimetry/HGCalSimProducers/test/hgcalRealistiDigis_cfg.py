import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9 as Era_Phase2
process = cms.Process("REALDIGI", Era_Phase2)


from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('standard')
options.register('geometry', 'Extended2026D94', VarParsing.multiplicity.singleton,  VarParsing.varType.string,  'geometry to use')
options.register('modules','Geometry/HGCalMapping/data/ModuleMaps/modulelocator_CEminus_V15p5.txt',mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells','Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells','Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.parseArguments()

if len(options.files)==0:
    options.files=['file:/eos/cms/store/group/dpg_hgcal/comm_hgcal/psilva/hackhathon/23234.103_TTbar_14TeV+2026D94Aging3000/step2.root']
    print(f'Using hackathon test files: {options.files}')

#set geometry/global tag
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.Geometry%sReco_cff'%options.geometry)
process.load('Configuration.Geometry.Geometry%s_cff'%options.geometry)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 500

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.files),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck")
                        )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )


#ESSources/Producers for the logical mapping
#indexers
process.load('Geometry.HGCalMapping.hgCalMappingIndexESSource_cfi')
process.hgCalMappingIndexESSource.modules = cms.FileInPath(options.modules)
process.hgCalMappingIndexESSource.si = cms.FileInPath(options.sicells)
process.hgCalMappingIndexESSource.sipm = cms.FileInPath(options.sipmcells)

process.load('Configuration.StandardSequences.Accelerators_cff')
process.hgCalMappingModuleESProducer = cms.ESProducer('hgcal::HGCalMappingModuleESProducer@alpaka',
                                                      filename=cms.FileInPath(options.modules),
                                                      moduleindexer=cms.ESInputTag('') )
process.hgCalMappingCellESProducer = cms.ESProducer('hgcal::HGCalMappingCellESProducer@alpaka',
                                                      filelist=cms.vstring(options.sicells,options.sipmcells),
                                                      cellindexer=cms.ESInputTag('') )

#realistic digis producer
process.hgCalDigiSoaFiller = cms.EDProducer('HGCalDigiSoaFiller@alpaka')
process.t = cms.Task(process.hgCalDigiSoaFiller)
process.p = cms.Path(process.t)

#output
process.output = cms.OutputModule('PoolOutputModule',
                                  fileName = cms.untracked.string(options.output),
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_*_*_REALDIGI'))

process.output_path = cms.EndPath(process.output)
