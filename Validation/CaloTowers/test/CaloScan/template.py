import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False

process.load("DQMServices.Core.DQMStore_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000) 
)
# Input source
process.source = cms.Source("PoolSource",
    firstEvent = cms.untracked.uint32(XXXXX), 
    fileNames = cms.untracked.vstring('file:mc.root') 
) 

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     splitLevel = cms.untracked.int32(0),
     fileName = cms.untracked.string("output.root")
)

process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001

process.HcalSimHitsAnalyser = cms.EDAnalyzer("HcalSimHitsValidation",
    outputFile = cms.untracked.string('HcalSimHitsValidation.root')
)   

process.hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile		      = cms.untracked.string('HcalDigisValidationRelVal.root'),
    digiLabel		      = cms.InputTag("hcalDigis"),
    zside		      = cms.untracked.string('*'),
    mode		      = cms.untracked.string('multi'),

    hcalselector	      = cms.untracked.string('all'),
    mc			      = cms.untracked.string('yes') # 'yes' for MC
)   

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
    eventype                  = cms.untracked.string('single'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('yes')  # default !
)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('yes')  # default!
)

#--- replace hbhereco with hbheprereco
delattr(process,"hbhereco")
process.hbhereco = process.hbheprereco.clone()
process.hcalLocalRecoSequence = cms.Sequence(process.hbhereco+process.hfreco+process.horeco)


#--- post-LS1 customization 
process.mix.digitizers.hcal.ho.photoelectronsToAnalog = cms.vdouble([4.0]*16)
process.mix.digitizers.hcal.ho.siPMCode = cms.int32(1)
process.mix.digitizers.hcal.ho.pixels = cms.int32(2500)
process.mix.digitizers.hcal.ho.doSiPMSmearing = cms.bool(False)
process.mix.digitizers.hcal.hf1.samplingFactor = cms.double(0.60)
process.mix.digitizers.hcal.hf2.samplingFactor = cms.double(0.60)
process.g4SimHits.HFShowerLibrary.FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v3.root'


#---------- PATH
process.g4SimHits.Generator.HepMCProductLabel = 'generator'
process.p = cms.Path(
 process.VtxSmeared * process.g4SimHits * process.mix *
 process.ecalDigiSequence * process.hcalDigiSequence *
 process.addPileupInfo *
 process.ecalPacker *
 process.esDigiToRaw *
 process.hcalRawData *
 process.rawDataCollector *
 process.ecalDigis *
 process.ecalPreshowerDigis *
 process.hcalDigis *
 process.calolocalreco *
 process.caloTowersRec *
 process.hcalnoise *
 process.HcalSimHitsAnalyser *
 process.hcalDigiAnalyzer *
 process.hcalTowerAnalyzer *
 process.hcalRecoAnalyzer *
 process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.FEVT)

