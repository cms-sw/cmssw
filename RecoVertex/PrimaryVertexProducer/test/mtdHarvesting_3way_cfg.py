import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
process = cms.Process('mtdHarvesting', Phase2C22I13M9)

input_file = "file:mtdValidation_3way.root"

print("=" * 60)
print("MTD 3-way harvesting (3D + 4D + GNN)")
print(f"Input: {input_file}")
print("=" * 60)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.Geometry.GeometryExtendedRun4D121Reco_cff")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(-1),
)

process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(input_file)
)

process.edmtome_step = cms.Path(process.EDMtoME)
process.dqmsave_step = cms.Path(process.DQMSaver)

process.load("Validation.MtdValidation.btlSimHitsPostProcessor_cfi")
process.load("Validation.MtdValidation.btlLocalRecoPostProcessor_cfi")
process.load("Validation.MtdValidation.MtdEleIsoPostProcessor_cfi")
process.load("Validation.MtdValidation.MtdTracksPostProcessor_cfi")
process.load("Validation.MtdValidation.Primary4DVertexPostProcessor_cfi")

process.Primary4DVertexPostProcessor3D = process.Primary4DVertexPostProcessor.clone(
    folder = cms.string('MTD/Vertices/3D/')
)
process.MtdTracksPostProcessor3D = process.MtdTracksPostProcessor.clone(
    folder = cms.string('MTD/Tracks/3D/')
)

process.Primary4DVertexPostProcessor.folder = cms.string('MTD/Vertices/4D/')
process.MtdTracksPostProcessor.folder = cms.string('MTD/Tracks/4D/')

process.Primary4DVertexPostProcessorGNN = process.Primary4DVertexPostProcessor.clone(
    folder = cms.string('MTD/Vertices/GNN/')
)
process.MtdTracksPostProcessorGNN = process.MtdTracksPostProcessor.clone(
    folder = cms.string('MTD/Tracks/GNN/')
)

process.harvesting = cms.Sequence(
    process.btlSimHitsPostProcessor +
    process.btlLocalRecoPostProcessor +
    process.MtdEleIsoPostProcessor +

    process.Primary4DVertexPostProcessor3D +
    process.MtdTracksPostProcessor3D +

    process.Primary4DVertexPostProcessor +
    process.MtdTracksPostProcessor +

    process.Primary4DVertexPostProcessorGNN +
    process.MtdTracksPostProcessorGNN
)

process.p = cms.Path(process.harvesting)

process.schedule = cms.Schedule(process.edmtome_step, process.p, process.dqmsave_step)

print("\nHarvesting post-processors:")
print("  - Primary4DVertexPostProcessor3D  -> MTD/Vertices/3D/")
print("  - Primary4DVertexPostProcessor    -> MTD/Vertices/4D/")
print("  - Primary4DVertexPostProcessorGNN -> MTD/Vertices/GNN/")
print("  - MtdTracksPostProcessor3D        -> MTD/Tracks/3D/")
print("  - MtdTracksPostProcessor          -> MTD/Tracks/4D/")
print("  - MtdTracksPostProcessorGNN       -> MTD/Tracks/GNN/")
