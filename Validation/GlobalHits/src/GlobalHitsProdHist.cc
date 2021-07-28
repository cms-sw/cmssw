/** \file GlobalHitsProdHist.cc
 *
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 */

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Validation/GlobalHits/interface/GlobalHitsProdHist.h"

GlobalHitsProdHist::GlobalHitsProdHist(const edm::ParameterSet &iPSet)
    : fName(""),
      verbosity(0),
      frequency(0),
      vtxunit(0),
      getAllProvenances(false),
      printProvenanceInfo(false),
      G4VtxSrc_Token_(consumes<edm::SimVertexContainer>((iPSet.getParameter<edm::InputTag>("G4VtxSrc")))),
      G4TrkSrc_Token_(consumes<edm::SimTrackContainer>(iPSet.getParameter<edm::InputTag>("G4TrkSrc"))),
      tGeomToken_(esConsumes()),
      cscGeomToken_(esConsumes()),
      dtGeomToken_(esConsumes()),
      rpcGeomToken_(esConsumes()),
      caloGeomToken_(esConsumes()),
      count(0) {
  std::string MsgLoggerCat = "GlobalHitsProdHist_GlobalHitsProdHist";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  edm::ParameterSet m_Prov = iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  // get Labels to use to extract information
  PxlBrlLowSrc_ = iPSet.getParameter<edm::InputTag>("PxlBrlLowSrc");
  PxlBrlHighSrc_ = iPSet.getParameter<edm::InputTag>("PxlBrlHighSrc");
  PxlFwdLowSrc_ = iPSet.getParameter<edm::InputTag>("PxlFwdLowSrc");
  PxlFwdHighSrc_ = iPSet.getParameter<edm::InputTag>("PxlFwdHighSrc");

  SiTIBLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTIBLowSrc");
  SiTIBHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTIBHighSrc");
  SiTOBLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTOBLowSrc");
  SiTOBHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTOBHighSrc");
  SiTIDLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTIDLowSrc");
  SiTIDHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTIDHighSrc");
  SiTECLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTECLowSrc");
  SiTECHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTECHighSrc");

  MuonCscSrc_ = iPSet.getParameter<edm::InputTag>("MuonCscSrc");
  MuonDtSrc_ = iPSet.getParameter<edm::InputTag>("MuonDtSrc");
  MuonRpcSrc_ = iPSet.getParameter<edm::InputTag>("MuonRpcSrc");

  ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");

  HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat)
        << "\n===============================\n"
        << "Initialized as EDProducer with parameter values:\n"
        << "    Name          = " << fName << "\n"
        << "    Verbosity     = " << verbosity << "\n"
        << "    Frequency     = " << frequency << "\n"
        << "    VtxUnit       = " << vtxunit << "\n"
        << "    GetProv       = " << getAllProvenances << "\n"
        << "    PrintProv     = " << printProvenanceInfo << "\n"
        << "    PxlBrlLowSrc  = " << PxlBrlLowSrc_.label() << ":" << PxlBrlLowSrc_.instance() << "\n"
        << "    PxlBrlHighSrc = " << PxlBrlHighSrc_.label() << ":" << PxlBrlHighSrc_.instance() << "\n"
        << "    PxlFwdLowSrc  = " << PxlFwdLowSrc_.label() << ":" << PxlBrlLowSrc_.instance() << "\n"
        << "    PxlFwdHighSrc = " << PxlFwdHighSrc_.label() << ":" << PxlBrlHighSrc_.instance() << "\n"
        << "    SiTIBLowSrc   = " << SiTIBLowSrc_.label() << ":" << SiTIBLowSrc_.instance() << "\n"
        << "    SiTIBHighSrc  = " << SiTIBHighSrc_.label() << ":" << SiTIBHighSrc_.instance() << "\n"
        << "    SiTOBLowSrc   = " << SiTOBLowSrc_.label() << ":" << SiTOBLowSrc_.instance() << "\n"
        << "    SiTOBHighSrc  = " << SiTOBHighSrc_.label() << ":" << SiTOBHighSrc_.instance() << "\n"
        << "    SiTIDLowSrc   = " << SiTIDLowSrc_.label() << ":" << SiTIDLowSrc_.instance() << "\n"
        << "    SiTIDHighSrc  = " << SiTIDHighSrc_.label() << ":" << SiTIDHighSrc_.instance() << "\n"
        << "    SiTECLowSrc   = " << SiTECLowSrc_.label() << ":" << SiTECLowSrc_.instance() << "\n"
        << "    SiTECHighSrc  = " << SiTECHighSrc_.label() << ":" << SiTECHighSrc_.instance() << "\n"
        << "    MuonCscSrc    = " << MuonCscSrc_.label() << ":" << MuonCscSrc_.instance() << "\n"
        << "    MuonDtSrc     = " << MuonDtSrc_.label() << ":" << MuonDtSrc_.instance() << "\n"
        << "    MuonRpcSrc    = " << MuonRpcSrc_.label() << ":" << MuonRpcSrc_.instance() << "\n"
        << "    ECalEBSrc     = " << ECalEBSrc_.label() << ":" << ECalEBSrc_.instance() << "\n"
        << "    ECalEESrc     = " << ECalEESrc_.label() << ":" << ECalEESrc_.instance() << "\n"
        << "    ECalESSrc     = " << ECalESSrc_.label() << ":" << ECalESSrc_.instance() << "\n"
        << "    HCalSrc       = " << HCalSrc_.label() << ":" << HCalSrc_.instance() << "\n"
        << "===============================\n";
  }

  // create histograms
  Char_t hname[200];
  Char_t htitle[200];

  // MCGeant
  sprintf(hname, "hMCRGP1");
  histName_.push_back(hname);
  sprintf(htitle, "RawGenParticles");
  hMCRGP[0] = new TH1F(hname, htitle, 100, 0., 5000.);
  sprintf(hname, "hMCRGP2");
  histName_.push_back(hname);
  hMCRGP[1] = new TH1F(hname, htitle, 100, 0., 500.);
  for (Int_t i = 0; i < 2; ++i) {
    hMCRGP[i]->GetXaxis()->SetTitle("Number of Raw Generated Particles");
    hMCRGP[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMCRGP[i]->GetName()] = hMCRGP[i];
  }

  sprintf(hname, "hMCG4Vtx1");
  histName_.push_back(hname);
  sprintf(htitle, "G4 Vertices");
  hMCG4Vtx[0] = new TH1F(hname, htitle, 100, 0., 50000.);
  sprintf(hname, "hMCG4Vtx2");
  histName_.push_back(hname);
  hMCG4Vtx[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);
  for (Int_t i = 0; i < 2; ++i) {
    hMCG4Vtx[i]->GetXaxis()->SetTitle("Number of Vertices");
    hMCG4Vtx[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMCG4Vtx[i]->GetName()] = hMCG4Vtx[i];
  }

  sprintf(hname, "hMCG4Trk1");
  histName_.push_back(hname);
  sprintf(htitle, "G4 Tracks");
  hMCG4Trk[0] = new TH1F(hname, htitle, 150, 0., 15000.);
  sprintf(hname, "hMCG4Trk2");
  histName_.push_back(hname);
  hMCG4Trk[1] = new TH1F(hname, htitle, 150, -0.5, 99.5);
  for (Int_t i = 0; i < 2; ++i) {
    hMCG4Trk[i]->GetXaxis()->SetTitle("Number of Tracks");
    hMCG4Trk[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMCG4Trk[i]->GetName()] = hMCG4Trk[i];
  }

  sprintf(hname, "hGeantVtxX1");
  histName_.push_back(hname);
  sprintf(htitle, "Geant vertex x/micrometer");
  hGeantVtxX[0] = new TH1F(hname, htitle, 100, -8000000., 8000000.);
  sprintf(hname, "hGeantVtxX2");
  histName_.push_back(hname);
  hGeantVtxX[1] = new TH1F(hname, htitle, 100, -50., 50.);
  for (Int_t i = 0; i < 2; ++i) {
    hGeantVtxX[i]->GetXaxis()->SetTitle("x of Vertex (um)");
    hGeantVtxX[i]->GetYaxis()->SetTitle("Count");
    histMap_[hGeantVtxX[i]->GetName()] = hGeantVtxX[i];
  }

  sprintf(hname, "hGeantVtxY1");
  histName_.push_back(hname);
  sprintf(htitle, "Geant vertex y/micrometer");
  hGeantVtxY[0] = new TH1F(hname, htitle, 100, -8000000, 8000000.);
  sprintf(hname, "hGeantVtxY2");
  histName_.push_back(hname);
  hGeantVtxY[1] = new TH1F(hname, htitle, 100, -50., 50.);
  for (Int_t i = 0; i < 2; ++i) {
    hGeantVtxY[i]->GetXaxis()->SetTitle("y of Vertex (um)");
    hGeantVtxY[i]->GetYaxis()->SetTitle("Count");
    histMap_[hGeantVtxY[i]->GetName()] = hGeantVtxY[i];
  }

  sprintf(hname, "hGeantVtxZ1");
  histName_.push_back(hname);
  sprintf(htitle, "Geant vertex z/millimeter");
  hGeantVtxZ[0] = new TH1F(hname, htitle, 100, -11000., 11000.);
  sprintf(hname, "hGeantVtxZ2");
  histName_.push_back(hname);
  hGeantVtxZ[1] = new TH1F(hname, htitle, 100, -250., 250.);
  for (Int_t i = 0; i < 2; ++i) {
    hGeantVtxZ[i]->GetXaxis()->SetTitle("z of Vertex (mm)");
    hGeantVtxZ[i]->GetYaxis()->SetTitle("Count");
    histMap_[hGeantVtxZ[i]->GetName()] = hGeantVtxZ[i];
  }

  sprintf(hname, "hGeantTrkPt");
  histName_.push_back(hname);
  sprintf(htitle, "Geant track pt/GeV");
  hGeantTrkPt = new TH1F(hname, htitle, 100, 0., 200.);
  hGeantTrkPt->GetXaxis()->SetTitle("pT of Track (GeV)");
  hGeantTrkPt->GetYaxis()->SetTitle("Count");
  histMap_[hGeantTrkPt->GetName()] = hGeantTrkPt;

  sprintf(hname, "hGeantTrkE");
  histName_.push_back(hname);
  sprintf(htitle, "Geant track E/GeV");
  hGeantTrkE = new TH1F(hname, htitle, 100, 0., 5000.);
  hGeantTrkE->GetXaxis()->SetTitle("E of Track (GeV)");
  hGeantTrkE->GetYaxis()->SetTitle("Count");
  histMap_[hGeantTrkE->GetName()] = hGeantTrkE;

  // ECal
  sprintf(hname, "hCaloEcal1");
  histName_.push_back(hname);
  sprintf(htitle, "Ecal hits");
  hCaloEcal[0] = new TH1F(hname, htitle, 100, 0., 10000.);
  sprintf(hname, "hCaloEcal2");
  histName_.push_back(hname);
  hCaloEcal[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);

  sprintf(hname, "hCaloEcalE1");
  histName_.push_back(hname);
  sprintf(htitle, "Ecal hits, energy/GeV");
  hCaloEcalE[0] = new TH1F(hname, htitle, 100, 0., 10.);
  sprintf(hname, "hCaloEcalE2");
  histName_.push_back(hname);
  hCaloEcalE[1] = new TH1F(hname, htitle, 100, 0., 0.1);

  sprintf(hname, "hCaloEcalToF1");
  histName_.push_back(hname);
  sprintf(htitle, "Ecal hits, ToF/ns");
  hCaloEcalToF[0] = new TH1F(hname, htitle, 100, 0., 1000.);
  sprintf(hname, "hCaloEcalToF2");
  histName_.push_back(hname);
  hCaloEcalToF[1] = new TH1F(hname, htitle, 100, 0., 100.);

  for (Int_t i = 0; i < 2; ++i) {
    hCaloEcal[i]->GetXaxis()->SetTitle("Number of Hits");
    hCaloEcal[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloEcal[i]->GetName()] = hCaloEcal[i];
    hCaloEcalE[i]->GetXaxis()->SetTitle("Energy of Hits (GeV)");
    hCaloEcalE[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloEcalE[i]->GetName()] = hCaloEcalE[i];
    hCaloEcalToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hCaloEcalToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloEcalToF[i]->GetName()] = hCaloEcalToF[i];
  }

  sprintf(hname, "hCaloEcalPhi");
  histName_.push_back(hname);
  sprintf(htitle, "Ecal hits, phi/rad");
  hCaloEcalPhi = new TH1F(hname, htitle, 100, -3.2, 3.2);
  hCaloEcalPhi->GetXaxis()->SetTitle("Phi of Hits (rad)");
  hCaloEcalPhi->GetYaxis()->SetTitle("Count");
  histMap_[hCaloEcalPhi->GetName()] = hCaloEcalPhi;

  sprintf(hname, "hCaloEcalEta");
  histName_.push_back(hname);
  sprintf(htitle, "Ecal hits, eta");
  hCaloEcalEta = new TH1F(hname, htitle, 100, -5.5, 5.5);
  hCaloEcalEta->GetXaxis()->SetTitle("Eta of Hits");
  hCaloEcalEta->GetYaxis()->SetTitle("Count");
  histMap_[hCaloEcalEta->GetName()] = hCaloEcalEta;

  sprintf(hname, "hCaloPreSh1");
  histName_.push_back(hname);
  sprintf(htitle, "PreSh hits");
  hCaloPreSh[0] = new TH1F(hname, htitle, 100, 0., 10000.);
  sprintf(hname, "hCaloPreSh2");
  histName_.push_back(hname);
  hCaloPreSh[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);

  sprintf(hname, "hCaloPreShE1");
  histName_.push_back(hname);
  sprintf(htitle, "PreSh hits, energy/GeV");
  hCaloPreShE[0] = new TH1F(hname, htitle, 100, 0., 10.);
  sprintf(hname, "hCaloPreShE2");
  histName_.push_back(hname);
  hCaloPreShE[1] = new TH1F(hname, htitle, 100, 0., 0.1);

  sprintf(hname, "hCaloPreShToF1");
  histName_.push_back(hname);
  sprintf(htitle, "PreSh hits, ToF/ns");
  hCaloPreShToF[0] = new TH1F(hname, htitle, 100, 0., 1000.);
  sprintf(hname, "hCaloPreShToF2");
  histName_.push_back(hname);
  hCaloPreShToF[1] = new TH1F(hname, htitle, 100, 0., 100.);

  for (Int_t i = 0; i < 2; ++i) {
    hCaloPreSh[i]->GetXaxis()->SetTitle("Number of Hits");
    hCaloPreSh[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloPreSh[i]->GetName()] = hCaloPreSh[i];
    hCaloPreShE[i]->GetXaxis()->SetTitle("Energy of Hits (GeV)");
    hCaloPreShE[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloPreShE[i]->GetName()] = hCaloPreShE[i];
    hCaloPreShToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hCaloPreShToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloPreShToF[i]->GetName()] = hCaloPreShToF[i];
  }

  sprintf(hname, "hCaloPreShPhi");
  histName_.push_back(hname);
  sprintf(htitle, "PreSh hits, phi/rad");
  hCaloPreShPhi = new TH1F(hname, htitle, 100, -3.2, 3.2);
  hCaloPreShPhi->GetXaxis()->SetTitle("Phi of Hits (rad)");
  hCaloPreShPhi->GetYaxis()->SetTitle("Count");
  histMap_[hCaloPreShPhi->GetName()] = hCaloPreShPhi;

  sprintf(hname, "hCaloPreShEta");
  histName_.push_back(hname);
  sprintf(htitle, "PreSh hits, eta");
  hCaloPreShEta = new TH1F(hname, htitle, 100, -5.5, 5.5);
  hCaloPreShEta->GetXaxis()->SetTitle("Eta of Hits");
  hCaloPreShEta->GetYaxis()->SetTitle("Count");
  histMap_[hCaloPreShEta->GetName()] = hCaloPreShEta;

  // Hcal
  sprintf(hname, "hCaloHcal1");
  histName_.push_back(hname);
  sprintf(htitle, "Hcal hits");
  hCaloHcal[0] = new TH1F(hname, htitle, 100, 0., 10000.);
  sprintf(hname, "hCaloHcal2");
  histName_.push_back(hname);
  hCaloHcal[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);

  sprintf(hname, "hCaloHcalE1");
  histName_.push_back(hname);
  sprintf(htitle, "Hcal hits, energy/GeV");
  hCaloHcalE[0] = new TH1F(hname, htitle, 100, 0., 10.);
  sprintf(hname, "hCaloHcalE2");
  histName_.push_back(hname);
  hCaloHcalE[1] = new TH1F(hname, htitle, 100, 0., 0.1);

  sprintf(hname, "hCaloHcalToF1");
  histName_.push_back(hname);
  sprintf(htitle, "Hcal hits, ToF/ns");
  hCaloHcalToF[0] = new TH1F(hname, htitle, 100, 0., 1000.);
  sprintf(hname, "hCaloHcalToF2");
  histName_.push_back(hname);
  hCaloHcalToF[1] = new TH1F(hname, htitle, 100, 0., 100.);

  for (Int_t i = 0; i < 2; ++i) {
    hCaloHcal[i]->GetXaxis()->SetTitle("Number of Hits");
    hCaloHcal[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloHcal[i]->GetName()] = hCaloHcal[i];
    hCaloHcalE[i]->GetXaxis()->SetTitle("Energy of Hits (GeV)");
    hCaloHcalE[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloHcalE[i]->GetName()] = hCaloHcalE[i];
    hCaloHcalToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hCaloHcalToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hCaloHcalToF[i]->GetName()] = hCaloHcalToF[i];
  }

  sprintf(hname, "hCaloHcalPhi");
  histName_.push_back(hname);
  sprintf(htitle, "Hcal hits, phi/rad");
  hCaloHcalPhi = new TH1F(hname, htitle, 100, -3.2, 3.2);
  hCaloHcalPhi->GetXaxis()->SetTitle("Phi of Hits (rad)");
  hCaloHcalPhi->GetYaxis()->SetTitle("Count");
  histMap_[hCaloHcalPhi->GetName()] = hCaloHcalPhi;

  sprintf(hname, "hCaloHcalEta");
  histName_.push_back(hname);
  sprintf(htitle, "Hcal hits, eta");
  hCaloHcalEta = new TH1F(hname, htitle, 100, -5.5, 5.5);
  hCaloHcalEta->GetXaxis()->SetTitle("Eta of Hits");
  hCaloHcalEta->GetYaxis()->SetTitle("Count");
  histMap_[hCaloHcalEta->GetName()] = hCaloHcalEta;

  // tracker
  sprintf(hname, "hTrackerPx1");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel hits");
  hTrackerPx[0] = new TH1F(hname, htitle, 100, 0., 10000.);
  sprintf(hname, "hTrackerPx2");
  histName_.push_back(hname);
  hTrackerPx[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);
  for (Int_t i = 0; i < 2; ++i) {
    hTrackerPx[i]->GetXaxis()->SetTitle("Number of Pixel Hits");
    hTrackerPx[i]->GetYaxis()->SetTitle("Count");
    histMap_[hTrackerPx[i]->GetName()] = hTrackerPx[i];
  }

  sprintf(hname, "hTrackerPxPhi");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel hits phi/rad");
  hTrackerPxPhi = new TH1F(hname, htitle, 100, -3.2, 3.2);
  hTrackerPxPhi->GetXaxis()->SetTitle("Phi of Hits (rad)");
  hTrackerPxPhi->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerPxPhi->GetName()] = hTrackerPxPhi;

  sprintf(hname, "hTrackerPxEta");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel hits eta");
  hTrackerPxEta = new TH1F(hname, htitle, 100, -3.5, 3.5);
  hTrackerPxEta->GetXaxis()->SetTitle("Eta of Hits");
  hTrackerPxEta->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerPxEta->GetName()] = hTrackerPxEta;

  sprintf(hname, "hTrackerPxBToF");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel barrel hits, ToF/ns");
  hTrackerPxBToF = new TH1F(hname, htitle, 100, 0., 40.);
  hTrackerPxBToF->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
  hTrackerPxBToF->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerPxBToF->GetName()] = hTrackerPxBToF;

  sprintf(hname, "hTrackerPxBR");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel barrel hits, R/cm");
  hTrackerPxBR = new TH1F(hname, htitle, 100, 0., 50.);
  hTrackerPxBR->GetXaxis()->SetTitle("R of Hits (cm)");
  hTrackerPxBR->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerPxBR->GetName()] = hTrackerPxBR;

  sprintf(hname, "hTrackerPxFToF");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel forward hits, ToF/ns");
  hTrackerPxFToF = new TH1F(hname, htitle, 100, 0., 50.);
  hTrackerPxFToF->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
  hTrackerPxFToF->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerPxFToF->GetName()] = hTrackerPxFToF;

  sprintf(hname, "hTrackerPxFZ");
  histName_.push_back(hname);
  sprintf(htitle, "Pixel forward hits, Z/cm");
  hTrackerPxFZ = new TH1F(hname, htitle, 200, -100., 100.);
  hTrackerPxFZ->GetXaxis()->SetTitle("Z of Hits (cm)");
  hTrackerPxFZ->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerPxFZ->GetName()] = hTrackerPxFZ;

  sprintf(hname, "hTrackerSi1");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon hits");
  hTrackerSi[0] = new TH1F(hname, htitle, 100, 0., 10000.);
  sprintf(hname, "hTrackerSi2");
  histName_.push_back(hname);
  hTrackerSi[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);
  for (Int_t i = 0; i < 2; ++i) {
    hTrackerSi[i]->GetXaxis()->SetTitle("Number of Silicon Hits");
    hTrackerSi[i]->GetYaxis()->SetTitle("Count");
    histMap_[hTrackerSi[i]->GetName()] = hTrackerSi[i];
  }

  sprintf(hname, "hTrackerSiPhi");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon hits phi/rad");
  hTrackerSiPhi = new TH1F(hname, htitle, 100, -3.2, 3.2);
  hTrackerSiPhi->GetXaxis()->SetTitle("Phi of Hits (rad)");
  hTrackerSiPhi->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerSiPhi->GetName()] = hTrackerSiPhi;

  sprintf(hname, "hTrackerSiEta");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon hits eta");
  hTrackerSiEta = new TH1F(hname, htitle, 100, -3.5, 3.5);
  hTrackerSiEta->GetXaxis()->SetTitle("Eta of Hits");
  hTrackerSiEta->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerSiEta->GetName()] = hTrackerSiEta;

  sprintf(hname, "hTrackerSiBToF");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon barrel hits, ToF/ns");
  hTrackerSiBToF = new TH1F(hname, htitle, 100, 0., 50.);
  hTrackerSiBToF->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
  hTrackerSiBToF->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerSiBToF->GetName()] = hTrackerSiBToF;

  sprintf(hname, "hTrackerSiBR");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon barrel hits, R/cm");
  hTrackerSiBR = new TH1F(hname, htitle, 100, 0., 200.);
  hTrackerSiBR->GetXaxis()->SetTitle("R of Hits (cm)");
  hTrackerSiBR->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerSiBR->GetName()] = hTrackerSiBR;

  sprintf(hname, "hTrackerSiFToF");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon forward hits, ToF/ns");
  hTrackerSiFToF = new TH1F(hname, htitle, 100, 0., 75.);
  hTrackerSiFToF->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
  hTrackerSiFToF->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerSiFToF->GetName()] = hTrackerSiFToF;

  sprintf(hname, "hTrackerSiFZ");
  histName_.push_back(hname);
  sprintf(htitle, "Silicon forward hits, Z/cm");
  hTrackerSiFZ = new TH1F(hname, htitle, 200, -300., 300.);
  hTrackerSiFZ->GetXaxis()->SetTitle("Z of Hits (cm)");
  hTrackerSiFZ->GetYaxis()->SetTitle("Count");
  histMap_[hTrackerSiFZ->GetName()] = hTrackerSiFZ;

  // muon
  sprintf(hname, "hMuon1");
  histName_.push_back(hname);
  sprintf(htitle, "Muon hits");
  hMuon[0] = new TH1F(hname, htitle, 100, 0., 10000.);
  sprintf(hname, "hMuon2");
  histName_.push_back(hname);
  hMuon[1] = new TH1F(hname, htitle, 100, -0.5, 99.5);
  for (Int_t i = 0; i < 2; ++i) {
    hMuon[i]->GetXaxis()->SetTitle("Number of Muon Hits");
    hMuon[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMuon[i]->GetName()] = hMuon[i];
  }

  sprintf(hname, "hMuonPhi");
  histName_.push_back(hname);
  sprintf(htitle, "Muon hits phi/rad");
  hMuonPhi = new TH1F(hname, htitle, 100, -3.2, 3.2);
  hMuonPhi->GetXaxis()->SetTitle("Phi of Hits (rad)");
  hMuonPhi->GetYaxis()->SetTitle("Count");
  histMap_[hMuonPhi->GetName()] = hMuonPhi;

  sprintf(hname, "hMuonEta");
  histName_.push_back(hname);
  sprintf(htitle, "Muon hits eta");
  hMuonEta = new TH1F(hname, htitle, 100, -3.5, 3.5);
  hMuonEta->GetXaxis()->SetTitle("Eta of Hits");
  hMuonEta->GetYaxis()->SetTitle("Count");
  histMap_[hMuonEta->GetName()] = hMuonEta;

  sprintf(hname, "hMuonCscToF1");
  histName_.push_back(hname);
  sprintf(htitle, "Muon CSC hits, ToF/ns");
  hMuonCscToF[0] = new TH1F(hname, htitle, 100, 0., 250.);
  sprintf(hname, "hMuonCscToF2");
  histName_.push_back(hname);
  hMuonCscToF[1] = new TH1F(hname, htitle, 100, 0., 50.);
  for (Int_t i = 0; i < 2; ++i) {
    hMuonCscToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hMuonCscToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMuonCscToF[i]->GetName()] = hMuonCscToF[i];
  }

  sprintf(hname, "hMuonCscZ");
  histName_.push_back(hname);
  sprintf(htitle, "Muon CSC hits, Z/cm");
  hMuonCscZ = new TH1F(hname, htitle, 200, -1500., 1500.);
  hMuonCscZ->GetXaxis()->SetTitle("Z of Hits (cm)");
  hMuonCscZ->GetYaxis()->SetTitle("Count");
  histMap_[hMuonCscZ->GetName()] = hMuonCscZ;

  sprintf(hname, "hMuonDtToF1");
  histName_.push_back(hname);
  sprintf(htitle, "Muon DT hits, ToF/ns");
  hMuonDtToF[0] = new TH1F(hname, htitle, 100, 0., 250.);
  sprintf(hname, "hMuonDtToF2");
  histName_.push_back(hname);
  hMuonDtToF[1] = new TH1F(hname, htitle, 100, 0., 50.);
  for (Int_t i = 0; i < 2; ++i) {
    hMuonDtToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hMuonDtToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMuonDtToF[i]->GetName()] = hMuonDtToF[i];
  }

  sprintf(hname, "hMuonDtR");
  histName_.push_back(hname);
  sprintf(htitle, "Muon DT hits, R/cm");
  hMuonDtR = new TH1F(hname, htitle, 100, 0., 1500.);
  hMuonDtR->GetXaxis()->SetTitle("R of Hits (cm)");
  hMuonDtR->GetYaxis()->SetTitle("Count");
  histMap_[hMuonDtR->GetName()] = hMuonDtR;

  sprintf(hname, "hMuonRpcFToF1");
  histName_.push_back(hname);
  sprintf(htitle, "Muon RPC forward hits, ToF/ns");
  hMuonRpcFToF[0] = new TH1F(hname, htitle, 100, 0., 250.);
  sprintf(hname, "hMuonRpcFToF2");
  histName_.push_back(hname);
  hMuonRpcFToF[1] = new TH1F(hname, htitle, 100, 0., 50.);
  for (Int_t i = 0; i < 2; ++i) {
    hMuonRpcFToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hMuonRpcFToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMuonRpcFToF[i]->GetName()] = hMuonRpcFToF[i];
  }

  sprintf(hname, "hMuonRpcFZ");
  histName_.push_back(hname);
  sprintf(htitle, "Muon RPC forward hits, Z/cm");
  hMuonRpcFZ = new TH1F(hname, htitle, 201, -1500., 1500.);
  hMuonRpcFZ->GetXaxis()->SetTitle("Z of Hits (cm)");
  hMuonRpcFZ->GetYaxis()->SetTitle("Count");
  histMap_[hMuonRpcFZ->GetName()] = hMuonRpcFZ;

  sprintf(hname, "hMuonRpcBToF1");
  histName_.push_back(hname);
  sprintf(htitle, "Muon RPC barrel hits, ToF/ns");
  hMuonRpcBToF[0] = new TH1F(hname, htitle, 100, 0., 250.);
  sprintf(hname, "hMuonRpcBToF2");
  histName_.push_back(hname);
  hMuonRpcBToF[1] = new TH1F(hname, htitle, 100, 0., 50.);
  for (Int_t i = 0; i < 2; ++i) {
    hMuonRpcBToF[i]->GetXaxis()->SetTitle("Time of Flight of Hits (ns)");
    hMuonRpcBToF[i]->GetYaxis()->SetTitle("Count");
    histMap_[hMuonRpcBToF[i]->GetName()] = hMuonRpcBToF[i];
  }

  sprintf(hname, "hMuonRpcBR");
  histName_.push_back(hname);
  sprintf(htitle, "Muon RPC barrel hits, R/cm");
  hMuonRpcBR = new TH1F(hname, htitle, 100, 0., 1500.);
  hMuonRpcBR->GetXaxis()->SetTitle("R of Hits (cm)");
  hMuonRpcBR->GetYaxis()->SetTitle("Count");
  histMap_[hMuonRpcBR->GetName()] = hMuonRpcBR;

  // create persistent objects
  for (std::size_t i = 0; i < histName_.size(); ++i) {
    produces<TH1F, edm::Transition::EndRun>(histName_[i]).setBranchAlias(histName_[i]);
  }
}

GlobalHitsProdHist::~GlobalHitsProdHist() {}

void GlobalHitsProdHist::beginJob() { return; }

void GlobalHitsProdHist::endJob() {
  std::string MsgLoggerCat = "GlobalHitsProdHist_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) << "Terminating having processed " << count << " events.";
  return;
}

void GlobalHitsProdHist::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalHitsProdHist_produce";

  // keep track of number of events processed
  ++count;

  // get event id information
  edm::RunNumber_t nrun = iEvent.id().run();
  edm::EventNumber_t nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat) << "Processing run " << nrun << ", event " << nevt << " (" << count << " events total)";
  } else if (verbosity == 0) {
    if (nevt % frequency == 0 || nevt == 1) {
      edm::LogInfo(MsgLoggerCat) << "Processing run " << nrun << ", event " << nevt << " (" << count
                                 << " events total)";
    }
  }

  // look at information available in the event
  if (getAllProvenances) {
    std::vector<const edm::StableProvenance *> AllProv;
    iEvent.getAllStableProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat) << "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity >= 0)) {
      TString eventout("\nProvenance info:\n");

      for (unsigned int i = 0; i < AllProv.size(); ++i) {
        eventout += "\n       ******************************";
        eventout += "\n       Module       : ";
        eventout += AllProv[i]->moduleLabel();
        eventout += "\n       ProductID    : ";
        eventout += AllProv[i]->productID().id();
        eventout += "\n       ClassName    : ";
        eventout += AllProv[i]->className();
        eventout += "\n       InstanceName : ";
        eventout += AllProv[i]->productInstanceName();
        eventout += "\n       BranchName   : ";
        eventout += AllProv[i]->branchName();
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }

  // call fill functions
  // gather G4MC information from event
  fillG4MC(iEvent);
  // gather Tracker information from event
  fillTrk(iEvent, iSetup);
  // gather muon information from event
  fillMuon(iEvent, iSetup);
  // gather Ecal information from event
  fillECal(iEvent, iSetup);
  // gather Hcal information from event
  fillHCal(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << "Done gathering data from event.";

  return;
}

void GlobalHitsProdHist::endRunProduce(edm::Run &iRun, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalHitsProdHist_endRun";

  TString eventout;
  TString eventoutw;
  bool warning = false;

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << "\nStoring histograms.";

  // store persistent objects
  std::map<std::string, TH1F *>::iterator iter;
  for (std::size_t i = 0; i < histName_.size(); ++i) {
    iter = histMap_.find(histName_[i]);
    if (iter != histMap_.end()) {
      std::unique_ptr<TH1F> hist1D(iter->second);
      eventout += "\n Storing histogram " + histName_[i];
      iRun.put(std::move(hist1D), histName_[i]);
    } else {
      warning = true;
      eventoutw += "\n Unable to find histogram with name " + histName_[i];
    }
  }

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
    if (warning)
      edm::LogWarning(MsgLoggerCat) << eventoutw << "\n";
  }
  return;
}

//==================fill and store functions================================
void GlobalHitsProdHist::fillG4MC(edm::Event &iEvent) {
  std::string MsgLoggerCat = "GlobalHitsProdHist_fillG4MC";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  //////////////////////
  // get MC information
  /////////////////////
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  std::vector<edm::Handle<edm::HepMCProduct>> AllHepMCEvt;
  iEvent.getManyByType(AllHepMCEvt);

  // loop through products and extract VtxSmearing if available. Any of them
  // should have the information needed
  for (unsigned int i = 0; i < AllHepMCEvt.size(); ++i) {
    HepMCEvt = AllHepMCEvt[i];
    if ((HepMCEvt.provenance()->branchDescription()).moduleLabel() == "generatorSmeared")
      break;
  }

  if (!HepMCEvt.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find HepMCProduct in event!";
    return;
  } else {
    eventout += "\n          Using HepMCProduct: ";
    eventout += (HepMCEvt.provenance()->branchDescription()).moduleLabel();
  }
  const HepMC::GenEvent *MCEvt = HepMCEvt->GetEvent();
  nRawGenPart = MCEvt->particles_size();

  if (verbosity > 1) {
    eventout += "\n          Number of Raw Particles collected:......... ";
    eventout += nRawGenPart;
  }

  if (hMCRGP[0])
    hMCRGP[0]->Fill((float)nRawGenPart);
  if (hMCRGP[1])
    hMCRGP[1]->Fill((float)nRawGenPart);

  ////////////////////////////
  // get G4Vertex information
  ////////////////////////////
  // convert unit stored in SimVertex to mm
  float unit = 0.;
  if (vtxunit == 0)
    unit = 1.;  // already in mm
  if (vtxunit == 1)
    unit = 10.;  // stored in cm, convert to mm

  edm::Handle<edm::SimVertexContainer> G4VtxContainer;
  iEvent.getByToken(G4VtxSrc_Token_, G4VtxContainer);
  if (!G4VtxContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find SimVertex in event!";
    return;
  }
  int i = 0;
  edm::SimVertexContainer::const_iterator itVtx;
  for (itVtx = G4VtxContainer->begin(); itVtx != G4VtxContainer->end(); ++itVtx) {
    ++i;

    const math::XYZTLorentzVector G4Vtx1(
        itVtx->position().x(), itVtx->position().y(), itVtx->position().z(), itVtx->position().e());

    double G4Vtx[4];
    G4Vtx1.GetCoordinates(G4Vtx);

    if (hGeantVtxX[0])
      hGeantVtxX[0]->Fill((G4Vtx[0] * unit) / micrometer);
    if (hGeantVtxX[1])
      hGeantVtxX[1]->Fill((G4Vtx[0] * unit) / micrometer);

    if (hGeantVtxY[0])
      hGeantVtxY[0]->Fill((G4Vtx[1] * unit) / micrometer);
    if (hGeantVtxY[1])
      hGeantVtxY[1]->Fill((G4Vtx[1] * unit) / micrometer);

    if (hGeantVtxZ[0])
      hGeantVtxZ[0]->Fill((G4Vtx[2] * unit) / millimeter);
    if (hGeantVtxZ[1])
      hGeantVtxZ[1]->Fill((G4Vtx[2] * unit) / millimeter);
  }

  if (verbosity > 1) {
    eventout += "\n          Number of G4Vertices collected:............ ";
    eventout += i;
  }

  if (hMCG4Vtx[0])
    hMCG4Vtx[0]->Fill((float)i);
  if (hMCG4Vtx[1])
    hMCG4Vtx[1]->Fill((float)i);

  ///////////////////////////
  // get G4Track information
  ///////////////////////////
  edm::Handle<edm::SimTrackContainer> G4TrkContainer;
  iEvent.getByToken(G4TrkSrc_Token_, G4TrkContainer);
  if (!G4TrkContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find SimTrack in event!";
    return;
  }
  i = 0;
  edm::SimTrackContainer::const_iterator itTrk;
  for (itTrk = G4TrkContainer->begin(); itTrk != G4TrkContainer->end(); ++itTrk) {
    ++i;

    const math::XYZTLorentzVector G4Trk1(
        itTrk->momentum().x(), itTrk->momentum().y(), itTrk->momentum().z(), itTrk->momentum().e());
    double G4Trk[4];
    G4Trk1.GetCoordinates(G4Trk);

    if (hGeantTrkPt)
      hGeantTrkPt->Fill(sqrt(G4Trk[0] * G4Trk[0] + G4Trk[1] * G4Trk[1]));
    if (hGeantTrkE)
      hGeantTrkE->Fill(G4Trk[3]);
  }

  if (verbosity > 1) {
    eventout += "\n          Number of G4Tracks collected:.............. ";
    eventout += i;
  }

  if (hMCG4Trk[0])
    hMCG4Trk[0]->Fill((float)i);
  if (hMCG4Trk[1])
    hMCG4Trk[1]->Fill((float)i);

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalHitsProdHist::fillTrk(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  nPxlHits = 0;
  std::string MsgLoggerCat = "GlobalHitsProdHist_fillTrk";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // access the tracker geometry
  const auto &theTrackerGeometry = iSetup.getHandle(tGeomToken_);
  if (!theTrackerGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerDigiGeometryRecord in event!";
    return;
  }
  const TrackerGeometry &theTracker(*theTrackerGeometry);

  // iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  ///////////////////////////////
  // get Pixel Barrel information
  ///////////////////////////////
  edm::PSimHitContainer thePxlBrlHits;
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlBrlLowContainer;
  iEvent.getByToken(PxlBrlLowSrc_Token_, PxlBrlLowContainer);
  if (!PxlBrlLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsPixelBarrelLowTof in event!";
    return;
  }
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlBrlHighContainer;
  iEvent.getByToken(PxlBrlHighSrc_Token_, PxlBrlHighContainer);
  if (!PxlBrlHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsPixelBarrelHighTof in event!";
    return;
  }
  // place both containers into new container
  thePxlBrlHits.insert(thePxlBrlHits.end(), PxlBrlLowContainer->begin(), PxlBrlLowContainer->end());
  thePxlBrlHits.insert(thePxlBrlHits.end(), PxlBrlHighContainer->begin(), PxlBrlHighContainer->end());

  // cycle through new container
  int i = 0, j = 0;
  for (itHit = thePxlBrlHits.begin(); itHit != thePxlBrlHits.end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && (subdetector == sdPxlBrl)) {
      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from PxlBrlHits for Hit " << i;
        continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      if (hTrackerPxBToF)
        hTrackerPxBToF->Fill(itHit->tof());
      if (hTrackerPxBR)
        hTrackerPxBR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
      if (hTrackerPxPhi)
        hTrackerPxPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (hTrackerPxEta)
        hTrackerPxEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "PxlBrl PSimHit " << i << " is expected to be (det,subdet) = (" << dTrk << ","
                                    << sdPxlBrl << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through PxlBrl Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Barrel Hits collected:..... ";
    eventout += j;
  }

  nPxlHits += j;

  /////////////////////////////////
  // get Pixel Forward information
  ////////////////////////////////
  edm::PSimHitContainer thePxlFwdHits;
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlFwdLowContainer;
  iEvent.getByToken(PxlFwdLowSrc_Token_, PxlFwdLowContainer);
  if (!PxlFwdLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsPixelEndcapLowTof in event!";
    return;
  }
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlFwdHighContainer;
  iEvent.getByToken(PxlFwdHighSrc_Token_, PxlFwdHighContainer);
  if (!PxlFwdHighContainer.isValid()) {
    edm::LogWarning("GlobalHitsProdHist_fillTrk") << "Unable to find TrackerHitsPixelEndcapHighTof in event!";
    return;
  }
  // place both containers into new container
  thePxlFwdHits.insert(thePxlFwdHits.end(), PxlFwdLowContainer->begin(), PxlFwdLowContainer->end());
  thePxlFwdHits.insert(thePxlFwdHits.end(), PxlFwdHighContainer->begin(), PxlFwdHighContainer->end());

  // cycle through new container
  i = 0;
  j = 0;
  for (itHit = thePxlFwdHits.begin(); itHit != thePxlFwdHits.end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && (subdetector == sdPxlFwd)) {
      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from PxlFwdHits for Hit " << i;
        ;
        continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      if (hTrackerPxFToF)
        hTrackerPxFToF->Fill(itHit->tof());
      if (hTrackerPxFZ)
        hTrackerPxFZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
      if (hTrackerPxPhi)
        hTrackerPxPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (hTrackerPxEta)
        hTrackerPxEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "PxlFwd PSimHit " << i << " is expected to be (det,subdet) = (" << dTrk << ","
                                    << sdPxlFwd << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Forward Hits collected:.... ";
    eventout += j;
  }

  nPxlHits += j;

  if (hTrackerPx[0])
    hTrackerPx[0]->Fill((float)nPxlHits);
  if (hTrackerPx[1])
    hTrackerPx[1]->Fill((float)nPxlHits);

  ///////////////////////////////////
  // get Silicon Barrel information
  //////////////////////////////////
  nSiHits = 0;
  edm::PSimHitContainer theSiBrlHits;
  // extract TIB low container
  edm::Handle<edm::PSimHitContainer> SiTIBLowContainer;
  iEvent.getByToken(SiTIBLowSrc_Token_, SiTIBLowContainer);
  if (!SiTIBLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTIBLowTof in event!";
    return;
  }
  // extract TIB high container
  edm::Handle<edm::PSimHitContainer> SiTIBHighContainer;
  iEvent.getByToken(SiTIBHighSrc_Token_, SiTIBHighContainer);
  if (!SiTIBHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTIBHighTof in event!";
    return;
  }
  // extract TOB low container
  edm::Handle<edm::PSimHitContainer> SiTOBLowContainer;
  iEvent.getByToken(SiTOBLowSrc_Token_, SiTOBLowContainer);
  if (!SiTOBLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTOBLowTof in event!";
    return;
  }
  // extract TOB high container
  edm::Handle<edm::PSimHitContainer> SiTOBHighContainer;
  iEvent.getByToken(SiTOBHighSrc_Token_, SiTOBHighContainer);
  if (!SiTOBHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTOBHighTof in event!";
    return;
  }
  // place all containers into new container
  theSiBrlHits.insert(theSiBrlHits.end(), SiTIBLowContainer->begin(), SiTIBLowContainer->end());
  theSiBrlHits.insert(theSiBrlHits.end(), SiTIBHighContainer->begin(), SiTIBHighContainer->end());
  theSiBrlHits.insert(theSiBrlHits.end(), SiTOBLowContainer->begin(), SiTOBLowContainer->end());
  theSiBrlHits.insert(theSiBrlHits.end(), SiTOBHighContainer->begin(), SiTOBHighContainer->end());

  // cycle through new container
  i = 0;
  j = 0;
  for (itHit = theSiBrlHits.begin(); itHit != theSiBrlHits.end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && ((subdetector == sdSiTIB) || (subdetector == sdSiTOB))) {
      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from SiBrlHits for Hit " << i;
        continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      if (hTrackerSiBToF)
        hTrackerSiBToF->Fill(itHit->tof());
      if (hTrackerSiBR)
        hTrackerSiBR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
      if (hTrackerSiPhi)
        hTrackerSiPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (hTrackerSiEta)
        hTrackerSiEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "SiBrl PSimHit " << i << " is expected to be (det,subdet) = (" << dTrk << ","
                                    << sdSiTIB << " || " << sdSiTOB << "); value returned is: (" << detector << ","
                                    << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through SiBrl Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Silicon Barrel Hits collected:... ";
    eventout += j;
  }

  nSiHits += j;

  ///////////////////////////////////
  // get Silicon Forward information
  ///////////////////////////////////
  edm::PSimHitContainer theSiFwdHits;
  // extract TID low container
  edm::Handle<edm::PSimHitContainer> SiTIDLowContainer;
  iEvent.getByToken(SiTIDLowSrc_Token_, SiTIDLowContainer);
  if (!SiTIDLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTIDLowTof in event!";
    return;
  }
  // extract TID high container
  edm::Handle<edm::PSimHitContainer> SiTIDHighContainer;
  iEvent.getByToken(SiTIDHighSrc_Token_, SiTIDHighContainer);
  if (!SiTIDHighContainer.isValid()) {
    edm::LogWarning("GlobalHitsProdHist_fillTrk") << "Unable to find TrackerHitsTIDHighTof in event!";
    return;
  }
  // extract TEC low container
  edm::Handle<edm::PSimHitContainer> SiTECLowContainer;
  iEvent.getByToken(SiTECLowSrc_Token_, SiTECLowContainer);
  if (!SiTECLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTECLowTof in event!";
    return;
  }
  // extract TEC high container
  edm::Handle<edm::PSimHitContainer> SiTECHighContainer;
  iEvent.getByToken(SiTECHighSrc_Token_, SiTECHighContainer);
  if (!SiTECHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerHitsTECHighTof in event!";
    return;
  }
  // place all containers into new container
  theSiFwdHits.insert(theSiFwdHits.end(), SiTIDLowContainer->begin(), SiTIDLowContainer->end());
  theSiFwdHits.insert(theSiFwdHits.end(), SiTIDHighContainer->begin(), SiTIDHighContainer->end());
  theSiFwdHits.insert(theSiFwdHits.end(), SiTECLowContainer->begin(), SiTECLowContainer->end());
  theSiFwdHits.insert(theSiFwdHits.end(), SiTECHighContainer->begin(), SiTECHighContainer->end());

  // cycle through container
  i = 0;
  j = 0;
  for (itHit = theSiFwdHits.begin(); itHit != theSiFwdHits.end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && ((subdetector == sdSiTID) || (subdetector == sdSiTEC))) {
      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from SiFwdHits Hit " << i;
        return;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      if (hTrackerSiFToF)
        hTrackerSiFToF->Fill(itHit->tof());
      if (hTrackerSiFZ)
        hTrackerSiFZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
      if (hTrackerSiPhi)
        hTrackerSiPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (hTrackerSiEta)
        hTrackerSiEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "SiFwd PSimHit " << i << " is expected to be (det,subdet) = (" << dTrk << ","
                                    << sdSiTOB << " || " << sdSiTEC << "); value returned is: (" << detector << ","
                                    << subdetector << ")";
      continue;
    }  // end check detector type
  }    // end loop through SiFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Silicon Forward Hits collected:.. ";
    eventout += j;
  }

  nSiHits += j;

  if (hTrackerSi[0])
    hTrackerSi[0]->Fill((float)nSiHits);
  if (hTrackerSi[1])
    hTrackerSi[1]->Fill((float)nSiHits);

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalHitsProdHist::fillMuon(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  nMuonHits = 0;
  std::string MsgLoggerCat = "GlobalHitsProdHist_fillMuon";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  ///////////////////////
  // access the CSC Muon
  ///////////////////////
  // access the CSC Muon geometry
  const auto &theCSCGeometry = iSetup.getHandle(cscGeomToken_);
  if (!theCSCGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find MuonGeometryRecord for the CSCGeometry in event!";
    return;
  }
  const CSCGeometry &theCSCMuon(*theCSCGeometry);

  // get Muon CSC information
  edm::Handle<edm::PSimHitContainer> MuonCSCContainer;
  iEvent.getByToken(MuonCscSrc_Token_, MuonCSCContainer);
  if (!MuonCSCContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find MuonCSCHits in event!";
    return;
  }

  // cycle through container
  int i = 0, j = 0;
  for (itHit = MuonCSCContainer->begin(); itHit != MuonCSCContainer->end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dMuon) && (subdetector == sdMuonCSC)) {
      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theCSCMuon.idToDetUnit(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from theCSCMuon for hit " << i;
        continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      if (hMuonCscToF[0])
        hMuonCscToF[0]->Fill(itHit->tof());
      if (hMuonCscToF[1])
        hMuonCscToF[1]->Fill(itHit->tof());
      if (hMuonCscZ)
        hMuonCscZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
      if (hMuonPhi)
        hMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (hMuonEta)
        hMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "MuonCsc PSimHit " << i << " is expected to be (det,subdet) = (" << dMuon << ","
                                    << sdMuonCSC << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through CSC Hits

  if (verbosity > 1) {
    eventout += "\n          Number of CSC muon Hits collected:......... ";
    eventout += j;
  }

  nMuonHits += j;

  /////////////////////
  // access the DT Muon
  /////////////////////
  // access the DT Muon geometry
  const auto &theDTGeometry = iSetup.getHandle(dtGeomToken_);
  if (!theDTGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find MuonGeometryRecord for the DTGeometry in event!";
    return;
  }
  const DTGeometry &theDTMuon(*theDTGeometry);

  // get Muon DT information
  edm::Handle<edm::PSimHitContainer> MuonDtContainer;
  iEvent.getByToken(MuonDtSrc_Token_, MuonDtContainer);
  if (!MuonDtContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find MuonDTHits in event!";
    return;
  }

  // cycle through container
  i = 0, j = 0;
  for (itHit = MuonDtContainer->begin(); itHit != MuonDtContainer->end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dMuon) && (subdetector == sdMuonDT)) {
      // CSC uses wires and layers rather than the full detID
      // get the wireId
      DTWireId wireId(itHit->detUnitId());

      // get the DTLayer from the geometry using the wireID
      const DTLayer *theDet = theDTMuon.layer(wireId.layerId());

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from theDtMuon for hit " << i;
        continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      if (hMuonDtToF[0])
        hMuonDtToF[0]->Fill(itHit->tof());
      if (hMuonDtToF[1])
        hMuonDtToF[1]->Fill(itHit->tof());
      if (hMuonDtR)
        hMuonDtR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
      if (hMuonPhi)
        hMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (hMuonEta)
        hMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "MuonDt PSimHit " << i << " is expected to be (det,subdet) = (" << dMuon << ","
                                    << sdMuonDT << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through DT Hits

  if (verbosity > 1) {
    eventout += "\n          Number of DT muon Hits collected:.......... ";
    eventout += j;
  }

  nMuonHits += j;

  // int RPCBrl = 0, RPCFwd = 0;
  ///////////////////////
  // access the RPC Muon
  ///////////////////////
  // access the RPC Muon geometry
  const auto &theRPCGeometry = iSetup.getHandle(rpcGeomToken_);
  if (!theRPCGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find MuonGeometryRecord for the RPCGeometry in event!";
    return;
  }
  const RPCGeometry &theRPCMuon(*theRPCGeometry);

  // get Muon RPC information
  edm::Handle<edm::PSimHitContainer> MuonRPCContainer;
  iEvent.getByToken(MuonRpcSrc_Token_, MuonRPCContainer);
  if (!MuonRPCContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find MuonRPCHits in event!";
    return;
  }

  // cycle through container
  i = 0, j = 0;
  int RPCBrl = 0, RPCFwd = 0;
  for (itHit = MuonRPCContainer->begin(); itHit != MuonRPCContainer->end(); ++itHit) {
    ++i;

    // create a DetID from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dMuon) && (subdetector == sdMuonRPC)) {
      // get an RPCDetID from the detUnitID
      RPCDetId RPCId(itHit->detUnitId());

      // find the region of the RPC hit
      int region = RPCId.region();

      // get the GeomDetUnit from the geometry using the RPCDetId
      const GeomDetUnit *theDet = theRPCMuon.idToDetUnit(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get GeomDetUnit from theRPCMuon for hit " << i;
        continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane &bSurface = theDet->surface();

      // gather necessary information
      if ((region == sdMuonRPCRgnFwdp) || (region == sdMuonRPCRgnFwdn)) {
        ++RPCFwd;

        if (hMuonRpcFToF[0])
          hMuonRpcFToF[0]->Fill(itHit->tof());
        if (hMuonRpcFToF[1])
          hMuonRpcFToF[1]->Fill(itHit->tof());
        if (hMuonRpcFZ)
          hMuonRpcFZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
        if (hMuonPhi)
          hMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
        if (hMuonEta)
          hMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

      } else if (region == sdMuonRPCRgnBrl) {
        ++RPCBrl;

        if (hMuonRpcBToF[0])
          hMuonRpcBToF[0]->Fill(itHit->tof());
        if (hMuonRpcBToF[1])
          hMuonRpcBToF[1]->Fill(itHit->tof());
        if (hMuonRpcBR)
          hMuonRpcBR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
        if (hMuonPhi)
          hMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
        if (hMuonEta)
          hMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

      } else {
        edm::LogWarning(MsgLoggerCat) << "Invalid region for RPC Muon hit" << i;
        continue;
      }  // end check of region
    } else {
      edm::LogWarning(MsgLoggerCat) << "MuonRpc PSimHit " << i << " is expected to be (det,subdet) = (" << dMuon << ","
                                    << sdMuonRPC << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through RPC Hits

  if (verbosity > 1) {
    eventout += "\n          Number of RPC muon Hits collected:......... ";
    eventout += j;
    eventout += "\n                    RPC Barrel muon Hits:............ ";
    eventout += RPCBrl;
    eventout += "\n                    RPC Forward muon Hits:........... ";
    eventout += RPCFwd;
  }

  nMuonHits += j;

  if (hMuon[0])
    hMuon[0]->Fill((float)nMuonHits);
  if (hMuon[1])
    hMuon[1]->Fill((float)nMuonHits);

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalHitsProdHist::fillECal(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalHitsProdHist_fillECal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // access the calorimeter geometry
  const auto &theCaloGeometry = iSetup.getHandle(caloGeomToken_);
  if (!theCaloGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find CaloGeometryRecord in event!";
    return;
  }
  const CaloGeometry &theCalo(*theCaloGeometry);

  // iterator to access containers
  edm::PCaloHitContainer::const_iterator itHit;

  ///////////////////////////////
  // get  ECal information
  ///////////////////////////////
  edm::PCaloHitContainer theECalHits;
  // extract EB container
  edm::Handle<edm::PCaloHitContainer> EBContainer;
  iEvent.getByToken(ECalEBSrc_Token_, EBContainer);
  if (!EBContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalHitsEB in event!";
    return;
  }
  // extract EE container
  edm::Handle<edm::PCaloHitContainer> EEContainer;
  iEvent.getByToken(ECalEESrc_Token_, EEContainer);
  if (!EEContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalHitsEE in event!";
    return;
  }
  // place both containers into new container
  theECalHits.insert(theECalHits.end(), EBContainer->begin(), EBContainer->end());
  theECalHits.insert(theECalHits.end(), EEContainer->begin(), EEContainer->end());

  // cycle through new container
  int i = 0, j = 0;
  for (itHit = theECalHits.begin(); itHit != theECalHits.end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->id());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dEcal) && ((subdetector == sdEcalBrl) || (subdetector == sdEcalFwd))) {
      // get the Cell geometry
      auto theDet = (theCalo.getSubdetectorGeometry(theDetUnitId))->getGeometry(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get CaloCellGeometry from ECalHits for Hit " << i;
        continue;
      }

      ++j;

      // get the global position of the cell
      const GlobalPoint &globalposition = theDet->getPosition();

      if (hCaloEcalE[0])
        hCaloEcalE[0]->Fill(itHit->energy());
      if (hCaloEcalE[1])
        hCaloEcalE[1]->Fill(itHit->energy());
      if (hCaloEcalToF[0])
        hCaloEcalToF[0]->Fill(itHit->time());
      if (hCaloEcalToF[1])
        hCaloEcalToF[1]->Fill(itHit->time());
      if (hCaloEcalPhi)
        hCaloEcalPhi->Fill(globalposition.phi());
      if (hCaloEcalEta)
        hCaloEcalEta->Fill(globalposition.eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "ECal PCaloHit " << i << " is expected to be (det,subdet) = (" << dEcal << ","
                                    << sdEcalBrl << " || " << sdEcalFwd << "); value returned is: (" << detector << ","
                                    << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through ECal Hits

  if (verbosity > 1) {
    eventout += "\n          Number of ECal Hits collected:............. ";
    eventout += j;
  }

  if (hCaloEcal[0])
    hCaloEcal[0]->Fill((float)j);
  if (hCaloEcal[1])
    hCaloEcal[1]->Fill((float)j);

  ////////////////////////////
  // Get Preshower information
  ////////////////////////////
  // extract PreShower container
  edm::Handle<edm::PCaloHitContainer> PreShContainer;
  iEvent.getByToken(ECalESSrc_Token_, PreShContainer);
  if (!PreShContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalHitsES in event!";
    return;
  }

  // cycle through container
  i = 0, j = 0;
  for (itHit = PreShContainer->begin(); itHit != PreShContainer->end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->id());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dEcal) && (subdetector == sdEcalPS)) {
      // get the Cell geometry
      auto theDet = (theCalo.getSubdetectorGeometry(theDetUnitId))->getGeometry(theDetUnitId);

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get CaloCellGeometry from PreShContainer for Hit " << i;
        continue;
      }

      ++j;

      // get the global position of the cell
      const GlobalPoint &globalposition = theDet->getPosition();

      if (hCaloPreShE[0])
        hCaloPreShE[0]->Fill(itHit->energy());
      if (hCaloPreShE[1])
        hCaloPreShE[1]->Fill(itHit->energy());
      if (hCaloPreShToF[0])
        hCaloPreShToF[0]->Fill(itHit->time());
      if (hCaloPreShToF[1])
        hCaloPreShToF[1]->Fill(itHit->time());
      if (hCaloPreShPhi)
        hCaloPreShPhi->Fill(globalposition.phi());
      if (hCaloPreShEta)
        hCaloPreShEta->Fill(globalposition.eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "PreSh PCaloHit " << i << " is expected to be (det,subdet) = (" << dEcal << ","
                                    << sdEcalPS << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through PreShower Hits

  if (verbosity > 1) {
    eventout += "\n          Number of PreSh Hits collected:............ ";
    eventout += j;
  }

  if (hCaloPreSh[0])
    hCaloPreSh[0]->Fill((float)j);
  if (hCaloPreSh[1])
    hCaloPreSh[1]->Fill((float)j);

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalHitsProdHist::fillHCal(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalHitsProdHist_fillHCal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // access the calorimeter geometry
  const auto &theCaloGeometry = iSetup.getHandle(caloGeomToken_);
  if (!theCaloGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find CaloGeometryRecord in event!";
    return;
  }
  const CaloGeometry &theCalo(*theCaloGeometry);

  // iterator to access containers
  edm::PCaloHitContainer::const_iterator itHit;

  ///////////////////////////////
  // get  HCal information
  ///////////////////////////////
  // extract HCal container
  edm::Handle<edm::PCaloHitContainer> HCalContainer;
  iEvent.getByToken(HCalSrc_Token_, HCalContainer);
  if (!HCalContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find HCalHits in event!";
    return;
  }

  // cycle through container
  int i = 0, j = 0;
  for (itHit = HCalContainer->begin(); itHit != HCalContainer->end(); ++itHit) {
    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->id());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dHcal) && ((subdetector == sdHcalBrl) || (subdetector == sdHcalEC) || (subdetector == sdHcalOut) ||
                                (subdetector == sdHcalFwd))) {
      // get the Cell geometry
      const HcalGeometry *theDet = dynamic_cast<const HcalGeometry *>(theCalo.getSubdetectorGeometry(theDetUnitId));

      if (!theDet) {
        edm::LogWarning(MsgLoggerCat) << "Unable to get HcalGeometry from HCalContainer for Hit " << i;
        continue;
      }

      ++j;

      // get the global position of the cell
      const GlobalPoint &globalposition = theDet->getPosition(theDetUnitId);

      if (hCaloHcalE[0])
        hCaloHcalE[0]->Fill(itHit->energy());
      if (hCaloHcalE[1])
        hCaloHcalE[1]->Fill(itHit->energy());
      if (hCaloHcalToF[0])
        hCaloHcalToF[0]->Fill(itHit->time());
      if (hCaloHcalToF[1])
        hCaloHcalToF[1]->Fill(itHit->time());
      if (hCaloHcalPhi)
        hCaloHcalPhi->Fill(globalposition.phi());
      if (hCaloHcalEta)
        hCaloHcalEta->Fill(globalposition.eta());

    } else {
      edm::LogWarning(MsgLoggerCat) << "HCal PCaloHit " << i << " is expected to be (det,subdet) = (" << dHcal << ","
                                    << sdHcalBrl << " || " << sdHcalEC << " || " << sdHcalOut << " || " << sdHcalFwd
                                    << "); value returned is: (" << detector << "," << subdetector << ")";
      continue;
    }  // end detector type check
  }    // end loop through HCal Hits

  if (verbosity > 1) {
    eventout += "\n          Number of HCal Hits collected:............. ";
    eventout += j;
  }

  if (hCaloHcal[0])
    hCaloHcal[0]->Fill((float)j);
  if (hCaloHcal[1])
    hCaloHcal[1]->Fill((float)j);

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}
