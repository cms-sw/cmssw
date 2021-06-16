import FWCore.ParameterSet.Config as cms
import six

process = cms.Process('SIMDIGIRECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTB160Module4XML_cfi')
process.load('Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalParametersInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')
process.load('Geometry.CaloEventSetup.HGCalTopology_cfi')
process.load('Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimG4CMS.HGCalTestBeam.DigiHGCalTB160_cff')
process.load('RecoLocalCalo.HGCalRecProducers.HGCalLocalRecoTestBeamSequence_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBAnalyzer_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('file:gensimdigireco.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('TBProt4MGenSimDigiReco.root')
                                   )

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_design', '')

process.generator = cms.EDProducer("FlatRandomEThetaGunProducer",
    AddAntiParticle = cms.bool(True),
    PGunParameters = cms.PSet(
        MinE = cms.double(120.0),
        MaxE = cms.double(120.0),
        MinTheta = cms.double(0.0),
        MaxTheta = cms.double(0.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        PartID = cms.vint32(2212)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single muon E 100')
)
process.VtxSmeared.MeanX  = 0.0
process.VtxSmeared.SigmaX = 0.55
process.VtxSmeared.MeanY  = 0.0
process.VtxSmeared.SigmaY = 0.65
process.VtxSmeared.MeanZ  = -3500.0
process.VtxSmeared.SigmaZ = 0
process.HGCalUncalibRecHit.HGCHEFConfig.isSiFE = False
process.g4SimHits.OnlySDs = ['HGCalSensitiveDetector', 
                             'AHcalSensitiveDetector',
                             'CaloTrkProcessing',
                             'HGCSensitiveDetector',
                             'EcalTBH4BeamDetector',
                             'HGCalTB1601SensitiveDetector',
                             'HcalTB02SensitiveDetector',
                             'HcalTB06BeamDetector',
                             'EcalSensitiveDetector',
                             'HcalSensitiveDetector']

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.mix)
process.reconstruction_step = cms.Path(process.HGCalLocalRecoTestBeamSequence)
process.analysis_step = cms.Path(process.HGCalTBAnalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.reconstruction_step,process.analysis_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

for label, prod in six.iteritems(process.producers_()):
        if prod.type_() == "OscarMTProducer":
            # ugly hack
            prod.__dict__['_TypedParameterizable__type'] = "OscarProducer"
