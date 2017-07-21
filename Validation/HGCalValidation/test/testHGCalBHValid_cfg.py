import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("testHGCalRecoLocal",eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2023D3Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D3_cff')
process.load('Geometry.HcalCommonData.testPhase2GeometryFineReco_cff')
process.load('Geometry.HcalCommonData.testPhase2GeometryFine_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

### setup HGCal local reco
# get uncalibrechits with weights method
process.load("RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi")
process.HGCalUncalibRecHit.HGCEEdigiCollection  = 'mix:HGCDigisEE'
process.HGCalUncalibRecHit.HGCHEFdigiCollection = 'mix:HGCDigisHEfront'

# get rechits e.g. from the weights
process.load("RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi")
process.HGCalRecHit.HGCEEuncalibRecHitCollection  = 'HGCalUncalibRecHit:HGCEEUncalibRecHits'
process.HGCalRecHit.HGCHEFuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEFUncalibRecHits'


process.HGCalRecoLocal = cms.Sequence(process.HGCalUncalibRecHit +
                                      process.HGCalRecHit )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 5
# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(''),
    annotation = cms.untracked.string(''),
    name = cms.untracked.string('Applications')
)


# Output definitiond
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
#    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('file:testHGCalBHValid.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-RECO')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ValidHGCal = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
    ),
    categories = cms.untracked.vstring('HcalSim',
        'ValidHGCal'),
    destinations = cms.untracked.vstring('cout','cerr')
)

# Additional output definition
process.load('Validation.HGCalValidation.test.hgcBHValidation_cfi')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hgcBHValid.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )


# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(35.0),
        MinPt = cms.double(35.0),
        PartID = cms.vint32(13), #--->muon
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(1.2),
        MaxEta = cms.double(3.0)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single muon pt 35'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)


#Modified to produce hgceedigis
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

#Following Removes Mag Field
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.bField = cms.double(0.0)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.recotest_step = cms.Path(process.HGCalRecoLocal)
process.analysis_step = cms.Path(process.hgcalBHValidation)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.recotest_step,process.analysis_step)
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.digi2raw_step,process.recotest_step,process.analysis_step)
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.recotest_step,process.analysis_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

for label, prod in process.producers_().iteritems():
        if prod.type_() == "OscarMTProducer":
            # ugly hack
            prod.__dict__['_TypedParameterizable__type'] = "OscarProducer"
