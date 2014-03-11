# MODIFIED BY HAND from cmsDriver.py ouput:
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: rerun_postls1 -s RAW2DIGI,L1 -n 1 --conditions POSTLS162_V2::All --magField 38T_PostLS1 --geometry Extended2015 --customise=L1Trigger/Configuration/customise_l1EmulatorFromRaw --processName=L1EmulRaw --no_exec --datatier DIGI-RECO --eventcontent FEVTDEBUGHLT --mc --filein dummy.root--customise=L1Trigger/Configuration/customise_l1EmulatorFromRaw --processName=L1EmulRaw --no_exec
import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('analysis')
options.parseArguments()

process = cms.Process('L1EmulRaw')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("file:data/crab_mix_noPU_step2.root")
)

process.options = cms.untracked.PSet(

)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'POSTLS162_V2::All', '')


process.l1extraParticles = cms.EDProducer(
    "L1ExtraParticlesProd",
    muonSource = cms.InputTag("simGmtDigis"),
    etTotalSource = cms.InputTag("simGctDigis"),
    nonIsolatedEmSource = cms.InputTag("simGctDigis","nonIsoEm"),
    etMissSource = cms.InputTag("simGctDigis"),
    htMissSource = cms.InputTag("simGctDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("simGctDigis","forJets"),
    centralJetSource = cms.InputTag("simGctDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("simGctDigis","tauJets"),
    isolatedEmSource = cms.InputTag("simGctDigis","isoEm"),
    etHadSource = cms.InputTag("simGctDigis"),
    hfRingEtSumsSource = cms.InputTag("simGctDigis"),
    hfRingBitCountsSource = cms.InputTag("simGctDigis"),
    centralBxOnly = cms.bool(True),
    ignoreHtMiss = cms.bool(False)
    )
process.L1Extra = cms.Sequence(process.l1extraParticles)

process.demo = cms.EDAnalyzer(
    'DustyDemo',
    muonSource = cms.InputTag("l1extraParticles"),
    nonIsolatedEmSource = cms.InputTag("l1extraParticles","NonIsolated"),
    etMissSource = cms.InputTag("l1extraParticles","MET"),
    htMissSource = cms.InputTag("l1extraParticles","MHT"),
    forwardJetSource = cms.InputTag("l1extraParticles","Forward"),
    centralJetSource = cms.InputTag("l1extraParticles","Central"),
    tauJetSource = cms.InputTag("l1extraParticles","Tau"),
    hfRingsSource = cms.InputTag("l1extraParticles"),
    particleMapSource = cms.InputTag("l1extraParticleMap"),
    isolatedEmSource = cms.InputTag("l1extraParticles","Isolated")
    )

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('hrerunl1.root')
    )

# Additional output definition
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")


#
# OPTION 1A:  run trigger primitive generation on unpacked ECAL digis, carried-forward HCAL unsupressed digis.
#            => works on MC as produced by current recipe of Gaelle
#            => produces decent but not perfect agreement between L1Emul and re-L1Emul on-fly ( few %)

process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')

process.CaloTPG_SimL1Emulator = cms.Sequence(
#    process.CaloTriggerPrimitives +
    process.SimL1Emulator
    )

# To run from unsuppressed ECAL Digis if included:
process.simEcalTriggerPrimitiveDigis.Label = 'simEcalUnsuppressedDigis'
process.simEcalTriggerPrimitiveDigis.InstanceEB= cms.string('')
process.simEcalTriggerPrimitiveDigis.InstanceEE= cms.string('')

# To run from unpacked ECAL Digis:
#process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'

process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    cms.InputTag('simHcalUnsuppressedDigis'),
    cms.InputTag('simHcalUnsuppressedDigis')
    )

process.L1simulation_step = cms.Path(process.CaloTPG_SimL1Emulator)

#
# Option 1B:  carry-forward the ECAL and HCAL TP digis.
#            => Pefect agreement, but not included in Gaelles current keep list
#

#process.L1simulation_step = cms.Path(process.SimL1Emulator)


process.raw2digi_step = cms.Path(process.RawToDigi)
#process.analysis_step = cms.Path(process.L1Extra*process.dumpED*process.demo)
process.analysis_step = cms.Path(process.L1Extra*process.demo)
# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1simulation_step,process.analysis_step)
#process.schedule = cms.Schedule(process.L1simulation_step,process.analysis_step)




# ES Producers accessed by post-LS1 muon customizations:
process.load('CalibMuon.CSCCalibration.CSCIndexer_cfi')
process.load('CalibMuon.CSCCalibration.CSCChannelMapper_cfi')
process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")

# Customizations for post-LS1 muon L1 emulator:
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_L1Emulator
process = customise_csc_L1Emulator(process)

# Use muonDTDigis from RAW2DIGI:
process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'

# Use simMuonDTDigis if available:
#process.simDtTriggerPrimitiveDigis.digiTag = 'simMuonDTDigis'
