
import FWCore.ParameterSet.Config as cms

electronMcSignalStdBining = cms.PSet(
  Ptmax = cms.double(100.0), Pmax = cms.double(300.0),
  Eopmax = cms.double(5.0),  Eopmaxsht = cms.double(3.0),
  Nbinxyz = cms.int32(50),
  Nbineop2D = cms.int32(30),
  Nbinp = cms.int32(50),
  Nbineta2D = cms.int32(50),
  Nbinfhits = cms.int32(30), Fhitsmax = cms.double(30.0),
  Nbinlhits = cms.int32(5), Lhitsmax = cms.double(10.0),
  Nbinpteff = cms.int32(19),
  Nbinphi2D = cms.int32(32),
  Nbineta = cms.int32(50), Etamin = cms.double(-2.5), Etamax = cms.double(2.5),
  Nbinp2D = cms.int32(50),
  Nbindeta = cms.int32(100), Detamin = cms.double(-0.005), Detamax = cms.double(0.005),
  Nbinpt2D = cms.int32(50),
  Nbindetamatch = cms.int32(100), Nbindetamatch2D = cms.int32(50), Detamatchmin = cms.double(-0.05), Detamatchmax = cms.double(0.05),
  Nbinphi = cms.int32(64), Phimin = cms.double(-3.2), Phimax = cms.double(3.2),
  Nbindphimatch = cms.int32(100), Nbindphimatch2D = cms.int32(50), Dphimatchmin = cms.double(-0.2), Dphimatchmax = cms.double(0.2),
  Nbinpt = cms.int32(50),
  Nbindphi = cms.int32(100), Dphimin = cms.double(-0.01), Dphimax = cms.double(0.01),
  Nbineop = cms.int32(50),
  Nbinpoptrue = cms.int32(75), Poptruemin = cms.double(0.0), Poptruemax = cms.double(1.5),
  Nbinmee = cms.int32(100), Meemin = cms.double(0.0), Meemax = cms.double(150.),
  Nbinhoe = cms.int32(100), Hoemin = cms.double(0.0), Hoemax = cms.double(0.5)
)

electronMcSignalFineBining = cms.PSet(electronMcSignalStdBining)
electronMcSignalFineBining.Poptruemin = cms.double(0.3)
electronMcSignalFineBining.Poptruemax = cms.double(1.2)
electronMcSignalFineBining.Nbinxyz = cms.int32(200)
electronMcSignalFineBining.Nbineop2D = cms.int32(100)
electronMcSignalFineBining.Nbinp = cms.int32(300)
electronMcSignalFineBining.Nbineta2D = cms.int32(100)
electronMcSignalFineBining.Nbinpteff = cms.int32(190)
electronMcSignalFineBining.Nbinphi2D = cms.int32(128)
electronMcSignalFineBining.Nbindetamatch2D = cms.int32(100)
electronMcSignalFineBining.Nbineta = cms.int32(250)
electronMcSignalFineBining.Nbinp2D = cms.int32(100)
electronMcSignalFineBining.Nbindeta = cms.int32(300)
electronMcSignalFineBining.Nbinpt2D = cms.int32(100)
electronMcSignalFineBining.Nbindetamatch = cms.int32(300)
electronMcSignalFineBining.Nbinphi = cms.int32(128)
electronMcSignalFineBining.Nbindphimatch = cms.int32(300)
electronMcSignalFineBining.Nbinpt = cms.int32(300)
electronMcSignalFineBining.Nbindphimatch2D = cms.int32(100)
electronMcSignalFineBining.Nbindphi = cms.int32(300)
electronMcSignalFineBining.Nbineop = cms.int32(300)
electronMcSignalFineBining.Nbinpoptrue = cms.int32(450)
electronMcSignalFineBining.Nbinmee = cms.int32(300)
electronMcSignalFineBining.Nbinhoe = cms.int32(200)

electronMcSignalValidator = cms.EDAnalyzer("ElectronMcSignalValidator",
  electronCollection = cms.InputTag("gsfElectrons"),
  mcTruthCollection = cms.InputTag("genParticles"),
  readAOD = cms.bool(False),
  outputFile = cms.string(""),
  MaxPt = cms.double(100.0),
  DeltaR = cms.double(0.05),
  MatchingID = cms.vint32(11,-11),
  MatchingMotherID = cms.vint32(23,24,-24,32),
  MaxAbsEta = cms.double(2.5),
  histosCfg = cms.PSet(electronMcSignalStdBining)
)



