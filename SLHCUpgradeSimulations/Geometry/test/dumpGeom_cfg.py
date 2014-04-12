import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
# Stole the basics from Fireworks/Geometry/python/ 
# This will create a geometry root file that cmsShow can read in.
# You will need to addpkg 'Fireworks/Geometry' and 'scram b'
# You will need to copy cmsRecoGeom1.root into the Fireworks/Geometry/data/ directory for cmsShow to find it
# Run RecoFull_Fullsim_Phase1_cfg to reconstruct the events that you want to show
# cmsShow -g cmsRecoGeom1.root reco.root
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = 'DESIGN42_V17::All'
process.load("SLHCUpgradeSimulations.Geometry.Phase1_R30F12_cmsSimIdealGeometryXML_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R30F12_cff")
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_phase1_cff")

process.add_(cms.ESProducer("FWRecoGeometryESProducer"))

#Adding Timing service:
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level = cms.untracked.int32(1)
                              )

process.p = cms.Path(process.dump)

