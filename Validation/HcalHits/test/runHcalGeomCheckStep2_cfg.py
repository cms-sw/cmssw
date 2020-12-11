import FWCore.ParameterSet.Config as cms

process = cms.Process("Analysis")

# import of standard configurations
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Validation.HcalHits.hcalGeomCheck_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HcalValidation=dict()


process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step1.root',
#       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_1_patch1/RelValSingleElectronPt35Extended/GEN-SIM-RECO/91X_upgrade2023_realistic_v1_D17-v1/10000/10D95AC2-B14A-E711-BC4A-0CC47A7C3638.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.Timing = cms.Service("Timing")

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hcalGeomCheck.root'),
                                   closeFileFast = cms.untracked.bool(True)
                            )

# Path definitions
process.analysis_step = cms.Path(process.hcalGeomCheck)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)

process.hcalGeomCheck.verbosity = 0
