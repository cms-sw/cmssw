import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('standard')
options.register('GlobalTag', "", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "GlobalTag to use (default abort)")
options.register('Redigi'   , 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Rerun digi from sim (default true)")
options.register('SLHC'   , 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Run SLHC mode (default false) doesn't do anything yet.")

options.parseArguments()
print options

### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load("SimCalorimetry.Configuration.hcalDigiSequence_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
process.load("RecoLocalCalo.Configuration.hcalGlobalReco_cff")
process.load("RecoMET.METProducers.hcalnoiseinfoproducer_cfi")
process.load("RecoJets.Configuration.CaloTowersRec_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("EventFilter.RawDataCollector.rawDataCollector_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['startup']
if(options.GlobalTag == ""):
    print("\n==================================================================================================")
    print("| No global tag set.  Please use option GlobalTag=XXXXXX::All to set the approperite global tag. |")
    print("==================================================================================================\n")
    exit()
process.GlobalTag.globaltag = cms.string(options.GlobalTag)


process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")


#process.load("DQMServices.Core.DQM_cfg")
#process.DQM.collectorHost = ''

process.load("DQMServices.Core.DQMStore_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) 
)
# Input source
process.source = cms.Source("PoolSource",
#    noEventSort = cms.untracked.bool(True),
#    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/0A15667E-7B81-E411-A10B-0025905A6068.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/161A8A1B-7781-E411-867C-00259059649C.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/2653E680-7B81-E411-92BB-0025905A48B2.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/8245EF1E-7781-E411-919E-0025905B85AA.root'
    ) ,
    secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/029B8EF8-5581-E411-B6D7-0025905B85AE.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/16A64AB2-5781-E411-A7C6-0025905B85AA.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/383DFC62-5581-E411-A822-0025905B85E8.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/44733F41-5681-E411-ADD1-002618943966.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/7859B8B3-5581-E411-BC59-0025905B85AE.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/94DD3149-5681-E411-8BA5-0025905A6082.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/AE6D3B5F-5581-E411-934A-0025905A6090.root',
       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/D2190515-5681-E411-B04F-002590596484.root'
    )
)

if(options.Redigi):
    process.source.inputCommands = cms.untracked.vstring('keep *',
                                         'drop *_MEtoEDMConverter_*_*',
                                         'drop HBHEDataFramesSorted_*__*',
                                         'drop HODataFramesSorted_*__*',
                                         'drop HFDataFramesSorted_*__*',
                                         'drop *_MEtoEDMConverter_*_*'
                                         )


process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     splitLevel = cms.untracked.int32(0),
     fileName = cms.untracked.string("output.root")
)

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

#--- EventFilter/HcalRawToDigi/python/HcalRawToDigi_cfi.py
process.hcalDigis.UnpackZDC =  cms.untracked.bool(False)

process.hcalDigis.InputLabel = 'rawDataCollector'

#--- replace hbhereco with hbheprereco
delattr(process,"hbhereco")
process.hbhereco = process.hbheprereco.clone()
process.hcalLocalRecoSequence = cms.Sequence(process.hbhereco+process.hfreco+process.horeco)


#---  Mixing:  ECAL and HCAL
process.mix.digitizers = cms.PSet(
#      ecal = cms.PSet(
#        process.ecalDigitizer
#      ),
      hcal = cms.PSet(
        process.hcalDigitizer
      )
)
process.simCastorDigis = cms.EDAlias()
process.simSiPixelDigis = cms.EDAlias()
process.simSiStripDigis = cms.EDAlias()
process.simEcalUnsuppressedDigis = cms.EDAlias()

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

if(options.Redigi):
    process.p = cms.Path(
        process.mix *
        process.hcalDigiSequence *
        process.hcalRawData *
        process.rawDataCollector *
        process.hcalDigis *
        process.hcalLocalRecoSequence *
        process.caloTowersRec *
        process.hcalnoise *
        process.hcalDigiAnalyzer *
        process.hcalTowerAnalyzer *
        process.hcalRecoAnalyzer *
        process.dqmSaver
        )
else:
    process.p = cms.Path(
        process.hcalDigiSequence *
        process.hcalRawData *
        process.rawDataCollector *
        process.hcalDigis *
        process.hcalLocalRecoSequence *
        process.caloTowersRec *
        process.hcalnoise *
        process.hcalDigiAnalyzer *
        process.hcalTowerAnalyzer *
        process.hcalRecoAnalyzer *
        process.dqmSaver

        )

#process.outpath = cms.EndPath(process.FEVT)

# Customization

from SLHCUpgradeSimulations.Configuration.postLS1Customs import customise_Digi
customise_Digi(process)
