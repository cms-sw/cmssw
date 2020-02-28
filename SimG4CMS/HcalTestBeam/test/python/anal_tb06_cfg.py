import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("SimG4CMS.HcalTestBeam.TB2006GeometryXML_cfi")

from SimG4CMS.HcalTestBeam.TB2006Analysis_cfi import *
process = testbeam2006(process)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('tb_pi_30gev.root')
)

#process.MessageLogger.destinations = cms.untracked.vstring("tb_pi_30gev.txt")

process.common_beam_direction_parameters.MinE = cms.double(30.0)
process.common_beam_direction_parameters.MaxE = cms.double(30.0)
process.common_beam_direction_parameters.PartID = cms.vint32(-211)

process.generator.PGunParameters.MinE = process.common_beam_direction_parameters.MinE
process.generator.PGunParameters.MaxE = process.common_beam_direction_parameters.MaxE
process.generator.PGunParameters.PartID = process.common_beam_direction_parameters.PartID

process.VtxSmeared.MinE = process.common_beam_direction_parameters.MinE
process.VtxSmeared.MaxE = process.common_beam_direction_parameters.MaxE
process.VtxSmeared.PartID = process.common_beam_direction_parameters.PartID

process.testbeam.MinE = process.common_beam_direction_parameters.MinE
process.testbeam.MaxE = process.common_beam_direction_parameters.MaxE
process.testbeam.PartID = process.common_beam_direction_parameters.PartID

process.testbeam.TestBeamAnalysis.EcalFactor = cms.double(1.)
process.testbeam.TestBeamAnalysis.HcalFactor = cms.double(100.)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

#process.Timing = cms.Service("Timing")

process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
