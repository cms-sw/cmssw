import FWCore.ParameterSet.Config as cms

process = cms.Process("GSD")




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   services
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   detector configuration
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.load('SLHCUpgradeSimulations.Geometry.Phase1_R30F12_HCal_cmsSimIdealGeometryXML_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.GlobalTag.globaltag = 'DESIGN42_V17::All'
process.load("SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R30F12_cff")



process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
#process.load("Configuration.StandardSequences.VtxSmearedBetafuncNominalCollision_cff")
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')





### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   event setup
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Sim_cff")

#process.load("Configuration.StandardSequences.Digi_cff")
process.load('SLHCUpgradeSimulations.Geometry.Digi_Phase1_R30F12_HCal_cff')

process.load('SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_HCal_cff')
process.mix.input.nbPileupEvents = cms.PSet(
            averageNumber = cms.double(0.0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

energy = 100. # in GeV

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer(
    "FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13, 211, 15, 11, 22),
 
        MinEta = cms.double(-2.0), ## in radians
        MaxEta = cms.double(2.0), ## in radians

        MinPhi = cms.double(-3.14159265359), ## in radians
        MaxPhi = cms.double(3.14159265359), ## in radians

        MinE = cms.double(energy), # in GeV
        MaxE = cms.double(energy) # in GeV
        ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single particle E ' + str(energy)),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
    )

process.pgen.remove(process.genJetMET)




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   configure mix to not do muons or tracker
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
from SimGeneral.MixingModule.mixObjects_cfi import *
process.mix.mixObjects = cms.PSet( mixCH = cms.PSet( mixCaloHits ) )



### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   configure HCAL
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#from SimCalorimetry/HcalTrigPrimProducers/test/exampleUpgradeTPG_cfg.py
# use hardcoded HCAL conditions values
process.es_hardcode.toGet.extend(['Gains', 'Pedestals', 'PedestalWidths', 'QIEData', 'ElectronicsMap','ChannelQuality','RespCorrs','ZSThresholds','L1TriggerObjects','TimeCorrs','PFCorrs','LUTCorrs'])

# Use text file for the LutMetadata
process.es_ascii = cms.ESSource("HcalTextCalibrations",
                                input = cms.VPSet (
        cms.PSet (
            object = cms.string ('LutMetadata'),
            file = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/HcalLutMetadataLSB50.txt')
            )
        )
)

# ESPrefers
process.es_prefer_hcalAscii    = cms.ESPrefer("HcalTextCalibrations"    , "es_ascii")
process.es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")

#Set SLHC modes
process.es_hardcode.SLHCMode = cms.untracked.bool(True)
process.es_hardcode.H2Mode = cms.untracked.bool(False)




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   configure HCAL simhits and digis for upgrade
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
doRelabeling = True

#turn on test numbering
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EMV'
process.g4SimHits.HCalSD.TestNumberingScheme = doRelabeling

#turn off zero suppression hopefully ?
process.simHcalDigis.HBlevel = -1000
process.simHcalDigis.HElevel = -1000
process.simHcalDigis.HOlevel = -1000

#turn on SiPMs in HO
process.hcalSimParameters.ho.siPMCode = 1
process.hcalSimParameters.ho.pixels = cms.int32(2500)
process.hcalSimParameters.ho.photoelectronsToAnalog = cms.vdouble(3.0)

#turn on SiPMs in HB/HE
process.hcalSimParameters.hb.siPMCells = [1]
process.hcalSimParameters.hb.pixels = cms.int32(4500*4)
process.hcalSimParameters.hb.photoelectronsToAnalog = cms.vdouble(10.0)
process.hcalSimParameters.he.pixels = cms.int32(4500*4)
process.hcalSimParameters.he.photoelectronsToAnalog = cms.vdouble(10.0)

#turn on SLHC topology
process.HcalTopologyIdealEP.SLHCMode = cms.untracked.bool(True)

#turn on hit relabeling and set depth segmentation
process.simHcalUnsuppressedDigis.RelabelHits = cms.untracked.bool(doRelabeling)
process.simHcalUnsuppressedDigis.RelabelRules = cms.untracked.PSet(
    # Eta1 = cms.untracked.vint32(1,1,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4),
    Eta1 = cms.untracked.vint32(1,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,5,5),
    #Eta17 = cms.untracked.vint32(1,1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4)
    Eta17 = cms.untracked.vint32(1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,5,5,5,5,5)
    )




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   CALO upgrade TPGs
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.load("SimCalorimetry.HcalTrigPrimProducers.hcalupgradetpdigi_cff")
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff")





### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   Upgrade Level-1 Trigger
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#process.load("L1Trigger.Configuration.L1CaloEmulator_cff")
process.load("L1Trigger.L1ExtraFromDigis.l1extraParticles_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

process.L1CaloTriggerSetup.InputXMLFile=cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/setup.xml')
process.L1CaloTowerProducer.ECALDigis = cms.InputTag("simEcalTriggerPrimitiveDigis")
process.L1CaloTowerProducer.HCALDigis = cms.InputTag("simHcalUpgradeTriggerPrimitiveDigis")
process.L1CaloTowerProducer.UseUpgradeHCAL = cms.bool(True)




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   Analysis
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.TFileService = cms.Service( "TFileService" , fileName = cms.string("histograms.root") )
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysis_cfi")

process.analysisSequence = cms.Sequence(
    process.SLHCelectrons*
    process.SLHCisoElectrons*
    process.SLHCphotons*
    process.SLHCisoPhotons*
    process.SLHCTaus*
    process.SLHCisoTaus*
    process.SLHCjets
)


### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   Processing path
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.p = cms.Path(
    process.pgen
    +process.psim
    +process.randomEngineStateProducer
    +process.mix
    +process.simHcalUnsuppressedDigis+process.simHcalDigis
    +process.simHcalUpgradeTriggerPrimitiveDigis
    +process.simEcalUnsuppressedDigis
    +process.simEcalTriggerPrimitiveDigis
    +process.SLHCCaloTrigger
    +process.mcSequence
    +process.analysisSequence
)





### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   Event output
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
process.load("Configuration.EventContent.EventContent_cff")
process.o1 = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'keep *_*_*_*'
    ),
    fileName = cms.untracked.string('UpgradeHcal.root')
)
#process.outpath = cms.EndPath(process.o1)




### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
#   Random stuff
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
# special treatment in case of production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator*getattr(process,path)._seq
