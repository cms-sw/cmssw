import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process('SIM',Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimG4CMS.HGCalTestBeam.HGCalTB181Oct1XML_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTB181Oct0XML_cfi')
process.load('Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalParametersInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
#process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBCheckGunPosition_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBAnalyzer_cfi')
#process.load('SimG4CMS.HGCalTestBeam.hgcalTBMBCERN_cfi')
#process.load('SimG4CMS.HGCalTestBeam.hgcalTBMBCERNBeam_cfi')
#process.load('SimG4CMS.HGCalTestBeam.hgcalTBMBCERN18Oct0_cfi')
process.load('SimG4CMS.HGCalTestBeam.hgcalTBMBCERNBeam18Oct0_cfi')
#process.load('SimG4CMS.HGCalTestBeam.hgcalTBMBCERN18Oct1_cfi')
#process.load('SimG4CMS.HGCalTestBeam.hgcalTBMBCERNBeam18Oct1_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCSim=dict()


# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleElectronE1000_cfi nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:gensim.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.TFileService = cms.Service("TFileService",
#                                  fileName = cms.string('TBCERNMB.root')
#                                  fileName = cms.string('TBCERNBeamMB.root')
#                                  fileName = cms.string('TBCERNOct0MB.root')
                                   fileName = cms.string('TBCERNBeamOct0MB.root')
#                                  fileName = cms.string('TBCERNOct1MB.root')
#                                  fileName = cms.string('TBCERNBeamOct1MB.root')
                                   )

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDProducer("FlatRandomEThetaGunProducer",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        MinE = cms.double(99.99),
        MaxE = cms.double(100.01),
        MinTheta = cms.double(0.0),
        MaxTheta = cms.double(0.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
#	MinTheta = cms.double(.011837),
#	MaxTheta = cms.double(.011837),
#	MinPhi = cms.double(3.649887),
#	MaxPhi = cms.double(3.649887),
        PartID = cms.vint32(14)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single muon E 100')
)

process.VtxSmeared.MinZ = -800.0
process.VtxSmeared.MaxZ = -800.0
process.VtxSmeared.MinX = 0
process.VtxSmeared.MaxX =  0
process.VtxSmeared.MinY = 0
process.VtxSmeared.MaxY =  0
process.HGCalTBAnalyzer.doDigis = False
process.HGCalTBAnalyzer.doRecHits = False
process.g4SimHits.StackingAction.TrackNeutrino = True

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.analysis_step = cms.Path(process.HGCalTBCheckGunPostion*process.HGCalTBAnalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(
    process.generation_step,
    process.genfiltersummary_step,
    process.simulation_step,
    process.analysis_step,
    process.endjob_step,
    process.RAWSIMoutput_step
)

# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


#print process.dumpPython()
