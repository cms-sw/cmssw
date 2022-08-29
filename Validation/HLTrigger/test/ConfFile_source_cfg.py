import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTGenValSource")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Core.DQMStore_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(6000) )

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100


# using: CMSSW_12_3_0_pre4__fullsim_noPU_2021_14TeV-TTbar_14TeV-00001
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL18RECO/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/00000/B4A06248-D09E-314A-ACD7-F157B86109E6.root")
    fileNames = cms.untracked.vstring(
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/07f08321-b24c-4397-b019-18c8ba54696c.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/22cc5971-134c-4a49-94b6-a3f96de01d94.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/2d20a2a4-b411-4124-bf1d-93db155b76e8.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/5f762599-4ddb-4c5f-8975-0229b54cae07.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/7964789d-c81b-4927-abaf-73acbd202abc.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/9cfca190-28f6-43af-b300-e3af7dbbfdd2.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/d22883fe-df35-48f3-ad2b-dbad44c8eaa4.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/f8ee9482-41e8-4126-ae00-bf07ef019d66.root",
    "root://cmsxrootd.fnal.gov//store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v4-v1/2580000/fe617389-a652-418d-b24e-55ea0ccacd7e.root",
    )
)

ptBins=cms.vdouble(0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,90,95 , 100,105,110,115 ,120,125, 130, 135,140,145, 150)
ptBinsHT=cms.vdouble(0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300)
ptBinsJet=cms.vdouble(0, 100, 200, 300, 350, 375, 400, 425, 450, 475, 500, 550, 600, 700, 800, 900, 1000)
etaBins=cms.vdouble(-4,-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4)

etaCut=cms.PSet(
    rangeVar=cms.string("eta"),
    allowedRanges=cms.vstring("-2.4:2.4")
)
ptCut=cms.PSet(
    rangeVar=cms.string("pt"),
    allowedRanges=cms.vstring("40:9999")
)

process.HLTGenValSourceHT = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("AK4HT"),
    hltPathsToCheck = cms.vstring(
      "HLT_PFHT1050_v",
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsHT,
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)

process.HLTGenValSourceHT = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("AK8HT"),
    hltPathsToCheck = cms.vstring(
      "HLT_AK8PFHT800_TrimMass50"
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsHT,
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)


process.HLTGenValSourceMU = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("mu"),
    hltPathsToCheck = cms.vstring(
      "HLT_Mu50_v:absEtaCut=1.2",
      "HLT_IsoMu24_v"
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBins,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)

process.HLTGenValSourceELE = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("ele"),
    hltPathsToCheck = cms.vstring(
      "HLT_Ele35_WPTight_Gsf_v",
      "HLT_Ele35_WPTight_Gsf_v:bins=ptBinsJet",
      "HLT_Ele115_CaloIdVT_GsfTrkIdT_v:region=EB",
      "HLT_Ele115_CaloIdVT_GsfTrkIdT_v:region=EE",
      "HLT_Photon200_v"
    ),
    binnings = cms.VPSet(
        cms.PSet(
            name = cms.string("ptBinsJet"),
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsJet
        )
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBins,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)

process.HLTGenValSourceAK4 = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("AK4jet"),
    hltPathsToCheck = cms.vstring(
      "HLT_PFJet500",
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsJet,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)

process.HLTGenValSourceAK8 = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("AK8jet"),
    hltPathsToCheck = cms.vstring(
      "HLT_AK8PFJet500",
      "HLT_AK8PFJet400_TrimMass30",
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsJet,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)

process.HLTGenValSourceMET = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("MET"),
    hltPathsToCheck = cms.vstring(
      "HLT_PFMET120_PFMHT120_IDTight",
    ),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBins,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins,
        ),
    ),
)

process.p = cms.Path(
        process.HLTGenValSourceMU *
        process.HLTGenValSourceELE *
        process.HLTGenValSourceHT *
        process.HLTGenValSourceAK4 *
        process.HLTGenValSourceAK8
        #process.HLTGenValSourceMET
        )

# the harvester
process.harvester = DQMEDHarvester("HLTGenValClient",
    outputFileName = cms.untracked.string('output.root'),
    subDirs        = cms.untracked.vstring("HLTGenVal"),
)

process.outpath = cms.EndPath(process.harvester)
