###############################################################################
# Way to use this:
#   cmsRun protoSimValid_cfg.py geometry=D77 type=hgcalBHValidation
#
#   Options for geometry D49, D68, D77, D83, D84, D88, D92, D93
#               type hgcalBHValidation, hgcalSiliconValidation
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

############################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D49, D68, D77, D83, D84, D88, D92, D93")
options.register ('type',
                  "hgcalBHValidation",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: hgcalBHValidation, hgcalSiliconValidation")

### get and parse the command line arguments
options.parseArguments()

print(options)

############################################################
# Use the options

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('PROD',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD49.root'
    else:
        fileName = 'hgcBHValidD49.root'
elif (options.geometry == "D68"):
    from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
    process = cms.Process('PROD',Phase2C12)
    process.load('Configuration.Geometry.GeometryExtended2026D68_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D68Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD68.root'
    else:
        fileName = 'hgcBHValidD68.root'
elif (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD83.root'
    else:
        fileName = 'hgcBHValidD83.root'
elif (options.geometry == "D84"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D84_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D84Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD84.root'
    else:
        fileName = 'hgcBHValidD84.root'
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD88.root'
    else:
        fileName = 'hgcBHValidD88.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D92_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD92.root'
    else:
        fileName = 'hgcBHValidD92.root'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D93_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD93.root'
    else:
        fileName = 'hgcBHValidD93.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    if (options.type == "hgcalSiliconValidation"):
        fileName = 'hgcSilValidD77.root'
    else:
        fileName = 'hgcBHValidD77.root'

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_Fake2_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

# Input source
process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(35.0),
        MinPt = cms.double(35.0),
        PartID = cms.vint32(13), #--->muon
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(1.2),
        MaxEta = cms.double(3.0)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single muon pt 35'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(''),
    annotation = cms.untracked.string(''),
    name = cms.untracked.string('Applications')
)

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True) )

#Modified to produce hgceedigis
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
process.ProductionFilterSequence = cms.Sequence(process.generator)

#Following Removes Mag Field
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.bField = cms.double(0.0)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.localreco)
process.recosim_step = cms.Path(process.recosim)

if (options.type == "hgcalSiliconValidation"):
    process.load('Validation.HGCalValidation.hgcalSiliconValidation_cfi')
    process.analysis_step = cms.Path(process.hgcalSiliconAnalysisEE+process.hgcalSiliconAnalysisHEF)
else:
    process.load('Validation.HGCalValidation.hgcalBHValidation_cfi')
    process.analysis_step = cms.Path(process.hgcalBHAnalysis)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
				process.simulation_step,
				process.digitisation_step,
                                process.L1simulation_step,
                                process.L1TrackTrigger_step,
                                process.digi2raw_step,
                                process.raw2digi_step,
                                process.L1Reco_step,
                                process.reconstruction_step,
                                process.recosim_step,
                                process.analysis_step,
				)

# filter all path with the production filter sequence
for path in process.paths:
        if getattr(process,path)._seq is not None: getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq
