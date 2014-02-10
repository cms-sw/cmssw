import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMDIGI")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# customization of the process.pdigi sequence to add the GEM digitizer
from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only
process = customize_digi_addGEM_muon_only(process)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_sim.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
        'file:out_digi.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        #'drop CastorDataFramesSorted_simCastorDigis_*_GEMDIGI'
        # drop all CF stuff
        ##'drop *_mix_*_*',
        # drop tracker simhits
        ##'drop PSimHits_*_Tracker*_*',
        # drop calorimetry stuff
        ##'drop PCaloHits_*_*_*',
        # clean up simhits from other detectors
        ##'drop PSimHits_*_Totem*_*',
        ##'drop PSimHits_*_FP420*_*',
        ##'drop PSimHits_*_BSC*_*',
        # drop some not useful muon digis and links
        ##'drop *_*_MuonCSCStripDigi_*',
        ##'drop *_*_MuonCSCStripDigiSimLinks_*',
        #'drop *SimLink*_*_*_*',
        ##'drop *RandomEngineStates_*_*_*',
        ##'drop *_randomEngineStateProducer_*_*'
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('digi_step')
    )
)

process.digi_step    = cms.Path(process.pdigi)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.digi_step,
    process.endjob_step,
    process.out_step
)
