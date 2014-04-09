import FWCore.ParameterSet.Config as cms

electronMcSignalHistosCfg = cms.PSet(
  Nbinxyz = cms.int32(50),
  Nbinp = cms.int32(50), Nbinp2D = cms.int32(50), 
#  Pmax = cms.double(300.0),
  Pmax = cms.double(4500), 
  Nbinpt = cms.int32(50), Nbinpt2D = cms.int32(50), Nbinpteff = cms.int32(19),Ptmax = cms.double(1500.0),
  Nbinfhits = cms.int32(30), Fhitsmax = cms.double(30.0),
  Nbinlhits = cms.int32(5), Lhitsmax = cms.double(10.0),
  Nbineta = cms.int32(50), Nbineta2D = cms.int32(50),Etamin = cms.double(-2.5), Etamax = cms.double(2.5),
  Nbindeta = cms.int32(100), Detamin = cms.double(-0.005), Detamax = cms.double(0.005), 
  Nbindetamatch = cms.int32(100), Nbindetamatch2D = cms.int32(50), Detamatchmin = cms.double(-0.05), Detamatchmax = cms.double(0.05),
  Nbinphi = cms.int32(64), Nbinphi2D = cms.int32(32), Phimin = cms.double(-3.2), Phimax = cms.double(3.2),
  Nbindphi = cms.int32(100), Dphimin = cms.double(-0.01), Dphimax = cms.double(0.01),
  Nbindphimatch = cms.int32(100), Nbindphimatch2D = cms.int32(50), Dphimatchmin = cms.double(-0.2), Dphimatchmax = cms.double(0.2),
  Nbineop = cms.int32(50), Nbineop2D = cms.int32(30), 
#  Eopmax = cms.double(5.0), 
  Eopmax = cms.double(10),
  Eopmaxsht = cms.double(3.0),
  Nbinmee = cms.int32(100), Meemin = cms.double(0.0), Meemax = cms.double(150.),
  Nbinhoe = cms.int32(100), Hoemin = cms.double(0.0), Hoemax = cms.double(0.5),
  Nbinpoptrue = cms.int32(75), Poptruemin = cms.double(0.0), Poptruemax = cms.double(1.5),
  Nbinerror = cms.int32(30), Energyerrormax = cms.double(150.0)
)

electronMcSignalValidator = cms.EDAnalyzer("ElectronMcSignalValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtRunEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
    
  mcTruthCollection = cms.InputTag("genParticles"),
  electronCollection = cms.InputTag("gedGsfElectrons"),
  electronCoreCollection = cms.InputTag("gedGsfElectronCores"),
  electronTrackCollection = cms.InputTag("electronGsfTracks"),
  electronSeedCollection = cms.InputTag("electronMergedSeeds"),
  
  beamSpot = cms.InputTag("offlineBeamSpot"),
  readAOD = cms.bool(False),
  
  isoFromDepsTk03            = cms.InputTag("eleIsoFromDepsTk03"),
  isoFromDepsTk04            = cms.InputTag("eleIsoFromDepsTk04"),
  isoFromDepsEcalFull03      = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystalFull03"),
  isoFromDepsEcalFull04      = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystalFull04"),
  isoFromDepsEcalReduced03   = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystalReduced03"),
  isoFromDepsEcalReduced04   = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystalReduced04"),
  isoFromDepsHcal03          = cms.InputTag("eleIsoFromDepsHcalFromTowers03"),
  isoFromDepsHcal04          = cms.InputTag("eleIsoFromDepsHcalFromTowers04"),
  
  MaxPt = cms.double(1500.0),
  DeltaR = cms.double(0.05),
  MaxAbsEta = cms.double(2.5),
  MatchingID = cms.vint32(11,-11),
  MatchingMotherID = cms.vint32(23,24,-24,32),
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



