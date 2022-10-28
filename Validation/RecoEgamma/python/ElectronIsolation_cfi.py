import FWCore.ParameterSet.Config as cms
import PhysicsTools.IsolationAlgos.CITKPFIsolationSumProducer_cfi as _mod

ElectronIsolation = _mod.CITKPFIsolationSumProducer.clone(
    srcToIsolate = "slimmedElectrons",
    srcForIsolationCone = 'packedPFCandidates',
    isolationConeDefinitions = cms.VPSet(
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeEndcaps = cms.double(0.015),
                  VetoConeSizeBarrel = cms.double(0.0),
                  isolateAgainst = cms.string('h+'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeEndcaps = cms.double(0.0),
                  VetoConeSizeBarrel = cms.double(0.0),
                  isolateAgainst = cms.string('h0'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeEndcaps = cms.double(0.08),
                  VetoConeSizeBarrel = cms.double(0.0),
                  isolateAgainst = cms.string('gamma'),
                  miniAODVertexCodes = cms.vuint32(2,3) )
    )
)
