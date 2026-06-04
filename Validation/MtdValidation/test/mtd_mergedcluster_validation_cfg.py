import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# Set up command line argument parsing
options = VarParsing('analysis')

options.register('inputFile',
                 'file:step3.root',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "Input ROOT file")

# Parse command line arguments
options.parseArguments()

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process("MergedClusterValidation", Phase2C17I13M9)

# Load standard configurations
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryExtendedRun4D110Reco_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

process.MessageLogger.cerr.threshold = 'INFO'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFile)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

# Validation analyzer
process.mtdMergedClusterValidation = cms.EDProducer("MergedClusterValidation",
    mergedClusters = cms.InputTag("mtdMergedClusters", "FTLBarrel"),
    clusters = cms.InputTag("mtdClusters", "FTLBarrel"),
    simMergedClusters = cms.InputTag("mtdSimMergedClusterProducer"),
    simLayerClusters = cms.InputTag("mix", "MergedMtdTruthLC"),
    sim2tpAssociationMapTag = cms.InputTag("mtdSimLayerClusterToTPAssociation", ""),  
    r2sAssociationMapTag = cms.InputTag("mtdRecoClusterToSimLayerClusterAssociation", "")
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Path
process.p = cms.Path( 
    process.mtdMergedClusterValidation
)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.schedule = cms.Schedule(process.p, process.endjob_step, process.DQMoutput_step)

print("Running MTD MergedCluster Validation...")
print(f"Input file: {options.inputFile}")
print(f"Output file: {options.outputFile}")
print(f"Max events: {options.maxEvents}")
