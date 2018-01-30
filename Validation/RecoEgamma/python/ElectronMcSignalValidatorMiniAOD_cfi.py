import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.gedGsfElectronFinalizer_cfi import gedGsfElectrons as _gedGsfElectrons

electronMcSignalHistosCfg = cms.PSet(
  Nbinxyz = cms.int32(50),
  Nbinpt = cms.int32(50), Nbinpt2D = cms.int32(50), Nbinpteff = cms.int32(19),Ptmax = cms.double(100.0),
  Nbinfhits = cms.int32(30), Fhitsmax = cms.double(30.0),
  Nbineta = cms.int32(50), Nbineta2D = cms.int32(50),Etamin = cms.double(-2.5), Etamax = cms.double(2.5),
  Nbindeta = cms.int32(100), Detamin = cms.double(-0.005), Detamax = cms.double(0.005), 
  Nbindetamatch = cms.int32(100), Nbindetamatch2D = cms.int32(50), Detamatchmin = cms.double(-0.05), Detamatchmax = cms.double(0.05),
  Nbinphi = cms.int32(64), Nbinphi2D = cms.int32(32), Phimin = cms.double(-3.2), Phimax = cms.double(3.2),
  Nbindphi = cms.int32(100), Dphimin = cms.double(-0.01), Dphimax = cms.double(0.01),
  Nbindphimatch = cms.int32(100), Nbindphimatch2D = cms.int32(50), Dphimatchmin = cms.double(-0.2), Dphimatchmax = cms.double(0.2),
  Nbinmee = cms.int32(100), Meemin = cms.double(0.0), Meemax = cms.double(150.),
  Nbinhoe = cms.int32(100), Hoemin = cms.double(0.0), Hoemax = cms.double(0.5),
  Nbinpoptrue = cms.int32(75), Poptruemin = cms.double(0.0), Poptruemax = cms.double(1.5),
  EfficiencyFlag = cms.bool(True), StatOverflowFlag = cms.bool(False)
)

electronPFIsolationCfg = cms.PSet(    
    pfSumChargedHadronPtTmp = cms.InputTag("miniAODElectronIsolation", _gedGsfElectrons.pfIsolationValues.pfSumChargedHadronPt.getProductInstanceLabel()),
    pfSumNeutralHadronEtTmp = cms.InputTag("miniAODElectronIsolation", _gedGsfElectrons.pfIsolationValues.pfSumNeutralHadronEt.getProductInstanceLabel()), # 
    pfSumPhotonEtTmp = cms.InputTag("miniAODElectronIsolation", _gedGsfElectrons.pfIsolationValues.pfSumPhotonEt.getProductInstanceLabel()), #  
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
electronMcSignalValidatorMiniAOD = DQMEDAnalyzer('ElectronMcSignalValidatorMiniAOD',

    Verbosity = cms.untracked.int32(0),
    FinalStep = cms.string("AtJobEnd"),
    InputFile = cms.string(""),
    OutputFile = cms.string(""),
    InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD"), # 
    OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD"), # 

    mcTruthCollection = cms.InputTag("prunedGenParticles"),
    electrons = cms.InputTag("slimmedElectrons"),
    
    MaxPt = cms.double(100.0),
    DeltaR = cms.double(0.05),
    MaxAbsEta = cms.double(2.5),
    MatchingID = cms.vint32(11,-11),
    MatchingMotherID = cms.vint32(23,24,-24,32),
    histosCfg = cms.PSet(electronMcSignalHistosCfg),
    isolationCfg = cms.PSet(electronPFIsolationCfg),
)
