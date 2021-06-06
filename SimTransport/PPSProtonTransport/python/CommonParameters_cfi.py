import FWCore.ParameterSet.Config as cms

commonParameters = cms.PSet(
                HepMCProductLabel = cms.InputTag('generatorSmeared'),
                Verbosity = cms.bool(False),
                EtaCut     = cms.double(8.2),
                MomentumCut= cms.double(3000),
                PPSRegionStart_45 = cms.double(212.45),
                PPSRegionStart_56 = cms.double(212.45)
)

