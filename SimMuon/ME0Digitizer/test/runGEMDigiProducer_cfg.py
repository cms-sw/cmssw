
import FWCore.ParameterSet.Config as cms

process = cms.Process("ME0DIGI")

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

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

# ME0 digitizer
process.load('SimMuon.ME0Digitizer.muonME0Digis_cfi')

# customization of the process.pdigi sequence to add the ME0 digitizer
from SimMuon.ME0Digitizer.customizeME0Digi import *
#process = customize_digi_addME0(process)  # run all detectors digi
process = customize_digi_addME0_muon_only(process) # only muon+ME0 digi
#process = customize_digi_addME0_me0_only(process)  # only ME0 digi

runCSCforSLHC = True
if runCSCforSLHC:
    # upgrade CSC customizations
    from SLHCUpgradeSimulations.Configuration.muonCustoms import *
    process = unganged_me1a_geometry(process)
    process = digitizer_timing_pre3_median(process)

addCSCstubs = True
if addCSCstubs:
    # unganged local stubs emulator:
    process.load('L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigisPostLS1_cfi')
    process.simCscTriggerPrimitiveDigis = process.cscTriggerPrimitiveDigisPostLS1.clone()

addPileUp = False
if addPileUp:
    # list of MinBias files for pileup has to be provided
    ff = open('filelist_pileup.txt', "r")
    pu_files = ff.read().split('\n')
    ff.close()
    pu_files = filter(lambda x: x.endswith('.root'),  pu_files)

    process.mix.input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
             #### THIS IS AVERAGE PILEUP NUMBER THAT YOU NEED TO CHANGE
            averageNumber = cms.double(50)
        ),
        type = cms.string('poisson'),
        sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(*pu_files)
    )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#        'file:out_sim.root'
'root://eoscms//eos/cms/store/user/mileva/me0Test/singleMuPt1000_gen_sim_merged620slhc/gensimMuPt1000Merged.root'

    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
        'file:out_digi_sven.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        #'drop CastorDataFramesSorted_simCastorDigis_*_ME0DIGI'
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


process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

if addCSCstubs:
    process.digi_step    = cms.Path(process.pdigi * process.simCscTriggerPrimitiveDigis)
else:
    process.digi_step    = cms.Path(process.pdigi)

process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.digi_step,
    process.endjob_step,
    process.out_step
)
