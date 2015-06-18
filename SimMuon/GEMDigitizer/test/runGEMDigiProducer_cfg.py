import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMDIGI")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(-1) 
)

#process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet( 
    wantSummary = cms.untracked.bool(True) 
)

# customization of the process.pdigi sequence to add the GEM digitizer 
# choose between different options below:

### GEM Only ############
# from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_gem_only
# process = customize_digi_addGEM_gem_only(process)  # only GEM digi
#########################

### All Muon Only #######
from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only
process = customize_digi_addGEM_muon_only(process) 
# from SimMuon.GEMDigitizer.customizeGEMDigi import customize_d2r_addGEM_muon_only
# process = customize_d2r_addGEM_muon_only(process) 
# from SimMuon.GEMDigitizer.customizeGEMDigi import customize_r2d_addGEM_muon_only
# process = customize_r2d_addGEM_muon_only(process) 
#########################

### All Detectors #######
# process = customize_digi_addGEM(process)  # run all detectors digi
#########################

### Fix RPC Digitization ###
############################
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions 
process = fixRPCConditions(process)
############################

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/afs/cern.ch/work/a/archie/public/SingleMuPt100_GEN-SIM__CMSSW_75X.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_digi.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
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

#process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

process.digi_step     = cms.Path(process.pdigi)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step   = cms.Path(process.endOfProcess)
process.out_step      = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    process.digi_step,
    process.endjob_step,
    process.out_step
)

#file = open('runGEMDigiProducer.py','w')
#file.write(str(process.dumpPython()))
#file.close()

