import FWCore.ParameterSet.Config as cms

process = cms.Process("ValidationSimDigiChain")
#
# Master configuration for the magnetic field
process.load('MagneticField.Engine.uniformMagneticField_cfi')

# Geometry master configuration
#
process.load('Geometry.EcalTestBeam.TBH4GeometryXML_cfi')
process.load('Geometry.CaloEventSetup.CaloGeometry_cfi')

process.load("Configuration.StandardSequences.Services_cff")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.EventContent.EventContent_cff")

# re-create the CrossingFrame
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# ECAL validation sequences
#
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("Validation.EcalHits.ecalSimHitsValidationSequence_cff")
process.load("Validation.EcalDigis.ecalDigisValidationSequence_cff")
process.ecalDigisValidation.EBdigiCollection = cms.InputTag('simEcalUnsuppressedDigis')
process.ecalBarrelDigisValidation.EBdigiCollection = cms.InputTag('simEcalUnsuppressedDigis')
process.load("Validation.EcalRecHits.ecalRecHitsValidationSequence_cff")
process.ecalRecHitsValidation.EBuncalibrechitCollection = cms.InputTag('ecalTBSimWeightUncalibRecHit:EcalUncalibRecHitsEB')
process.ecalRecHitsValidation.EBrechitCollection = cms.InputTag('ecalTBSimRecHit:EcalRecHitsEB')
process.ecalBarrelRecHitsValidation.EBuncalibrechitCollection = cms.InputTag('ecalTBSimWeightUncalibRecHit:EcalUncalibRecHitsEB')
process.ecalBarrelRecHitsValidation.EBdigiCollection = cms.InputTag('simEcalUnsuppressedDigis')

process.ecalTBValidation = cms.EDProducer("EcalTBValidation",
                                          digiCollection = cms.string(''),
                                          digiProducer = cms.string('simEcalUnsuppressedDigis'),
                                          hitCollection = cms.string('EcalUncalibRecHitsEB'),
                                          hitProducer = cms.string('ecalTBSimWeightUncalibRecHit'),
                                          hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
                                          hodoRecInfoProducer = cms.string('ecalTBSimHodoscopeReconstructor'),
                                          tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
                                          tdcRecInfoProducer = cms.string('ecalTBSimTDCReconstructor'),
                                          eventHeaderCollection = cms.string(''),
                                          eventHeaderProducer = cms.string('SimEcalEventHeader'),
                                          data = cms.untracked.int32(1),
                                          xtalInBeam = cms.untracked.int32(248),
                                          rootfile = cms.untracked.string('EcalTBValidation.root'),
                                          verbose = cms.untracked.bool(True)
)                                          

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.ecalSimValid  = cms.Sequence( process.ecalSimHitsValidationSequence + process.ecalDigisValidationSequence + process.ecalRecHitsValidationSequence + process.ecalTBValidation + process.MEtoEDMConverter )	

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:H4TB_50GeV_default_QGSP_BERT_EMV.root')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTDEBUGEventContent,
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('EcalTBH4Validation.root')
)

process.p0 = cms.Path(process.mix)
process.p2 = cms.Path(process.ecalSimValid)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p0,process.p2,process.outpath)

process.mix.playback = True

# drop the plain root file outputs of all analyzers
# Note: all the validation "analyzers" are EDFilters!
for filter in (getattr(process,f) for f in process.filters_()):
    if hasattr(filter,"outputFile"):
        filter.outputFile=""
        #Catch the problem with valid_HB.root that uses OutputFile instead of outputFile
        if hasattr(filter,"OutputFile"):
            filter.OutputFile=""

