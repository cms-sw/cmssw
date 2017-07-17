import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#Magnetic Field
#
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Output of events, etc...
#
# Explicit note : since some histos/tree might be dumped directly,
#                 better NOT use PoolOutputModule !
# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet( ## but FwkJob category - those unlimitted
           limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('FwkJob'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    #	untracked vstring fileNames = {"file:single_muminus_pT0.6_eta0.1.root"}
    fileNames = cms.untracked.vstring('file:single_piminus_pT0.2_eta0.1.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.TrackerSD.EnergyThresholdForPersistencyInGeV = 0.01
process.g4SimHits.TrackerSD.EnergyThresholdForHistoryInGeV = 0.001
process.g4SimHits.SteppingAction.KillBeamPipe = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('MaterialBudgetAction'),
    MaterialBudgetAction = cms.PSet(
        # Decay
        storeDecay = cms.untracked.bool(True),
        HistosFile = cms.string('matbdg_Gigi.root'),
        AllStepsToTree = cms.bool(True),
        HistogramList = cms.string('Tracker'),
        # vstring SelectedVolumes = {"CMSE"}
        SelectedVolumes = cms.vstring('Tracker', 
            'BEAM'),
        TreeFile = cms.string('loop_Gigi.root'), ## is NOT requested

        # vstring SelectedVolumes = {"Tracker"}
        # vstring SelectedVolumes = {"PixelBarrel","PixelForwardZMinus","PixelForwardZPlus","TOB","TIB","TIDB","TIDF","TEC","TIBTIDServicesF","TIBTIDServicesB"}
        StopAfterProcess = cms.string('None'),
        # string TextFile = "matbdg_Gigi.txt"
        TextFile = cms.string('None'),
        EminDecayProd = cms.untracked.double(0.0) ## MeV

    )
))


