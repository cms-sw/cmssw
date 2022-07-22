// -*- C++ -*-
//
// Package:    HGCalValidation/HGCalCellHitSum
// Class:      HGCalCellHitSum
//
/**\class HGCalCellHitSum HGCalCellHitSum.cc Validation/HGCalValidation/test/HGCalCellHitSum.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Indranil Das
//         Created:  Wed, 25 Aug 2021 06:18:11 GMT
//
//

// system include files
#include <memory>
#include <vector>
#include <fstream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToModule.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"
#include "DataFormats/Math/interface/angle_units.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <TH1.h>
#include <TH2.h>
#include <TGraph.h>
#include <TMath.h>

using namespace angle_units::operators;

//
// class declaration
//

class HGCalCellHitSum : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  //Implemented following Validation/HGCalValidation/plugins/HGCalSimHitValidation.cc
  struct energysum {
    energysum() {
      etotal = 0;
      for (int i = 0; i < 6; ++i)
        eTime[i] = 0.;
    }
    double eTime[6], etotal;
  };

  struct waferinfo {
    waferinfo() { layer = u = v = type = -999; }
    int layer, u, v, type;
  };

  struct hitsinfo {
    hitsinfo() {
      x = y = z = phi = eta = trkpt = trketa = trkphi = 0.0;
      cell = cell2 = sector = sector2 = type = layer = pdg = charge = 0;
      hitid = nhits = 0;
      isMu = false;
    }
    double x, y, z, phi, eta, trkpt, trketa, trkphi;
    int cell, cell2, sector, sector2, type, layer, pdg, charge;
    unsigned int hitid, nhits;
    bool isMu;
  };

  explicit HGCalCellHitSum(const edm::ParameterSet &);
  ~HGCalCellHitSum() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override {}

  // ----------member data ---------------------------
  const edm::EDGetTokenT<edm::SimTrackContainer> tSimTrackContainer;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tSimCaloHitContainer;
  const std::string name_;
  const edm::FileInPath geometryFileName_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  TH1D *hCharge;
  TH1D *hChargeLowELoss;

  TH1D *hPt;
  TH1D *hPtNoGen;
  TH1D *hPtLowELoss;

  TH1D *hEta;
  TH1D *hEtaCell;
  TH1D *hEtaLowELoss;

  TH1D *hPhi;
  TH1D *hPhiCell;
  TH1D *hPhiLowELoss;

  TH1D *hPDG;
  TH1D *hPDGLowELoss;

  TH1D *hELossEE;
  TH1D *hELossEEF;
  TH1D *hELossEECN;
  TH1D *hELossEECK;
  TH1D *hELossHEF;
  TH1D *hELossHEFF;
  TH1D *hELossHEFCN;
  TH1D *hELossHEFCK;
  TH1D *hELossHEB;

  TH1D *hELossCSinBunchEE;
  TH1D *hELossCSinBunchEEF;
  TH1D *hELossCSinBunchEECN;
  TH1D *hELossCSinBunchEECK;
  TH1D *hELossCSinBunchHEF;
  TH1D *hELossCSinBunchHEFF;
  TH1D *hELossCSinBunchHEFCN;
  TH1D *hELossCSinBunchHEFCK;
  TH1D *hELossCSinBunchHEFCNFiltered;
  TH1D *hELossCSinBunchHEFCNNoise;

  TH1D *hELossCSmissedEE;
  TH1D *hELossCSmissedEEF;
  TH1D *hELossCSmissedEECN;
  TH1D *hELossCSmissedEECK;
  TH1D *hELossCSmissedHEF;
  TH1D *hELossCSmissedHEFF;
  TH1D *hELossCSmissedHEFCN;
  TH1D *hELossCSmissedHEFCK;

  TH1D *hELossCSMaxEE;
  TH1D *hELossCSMaxEEF;
  TH1D *hELossCSMaxEECN;
  TH1D *hELossCSMaxEECK;
  TH1D *hELossCSMaxHEF;
  TH1D *hELossCSMaxHEFF;
  TH1D *hELossCSMaxHEFCN;
  TH1D *hELossCSMaxHEFCK;

  TH1D *hHxELossCSMaxF;
  TH1D *hHxELossCSMaxCN;
  TH1D *hHxELossCSMaxCK;
  TH1D *hNHxELossCSMaxF;
  TH1D *hNHxELossCSMaxCN;
  TH1D *hNHxELossCSMaxCK;

  std::vector<TH1D *> hELossDQMEqV;
  std::vector<TH1D *> hELossLayer;

  // TH2D *hYZhits;
  std::vector<TH2D *> hXYhits;
  std::vector<TH2D *> hXYhitsF;
  std::vector<TH2D *> hXYhitsCN;
  std::vector<TH2D *> hXYhitsCK;
  std::vector<TH2D *> hXYhitsB;
  std::vector<TH2D *> hXYhitsLELCN;
  std::vector<TH2D *> hXYhitsHELCN;
  std::vector<TH2D *> hXYhitsLELCK;
  std::vector<TH2D *> hXYhitsHELCK;
  std::vector<TH2D *> hNHxXYhitsF;
  std::vector<TH2D *> hNHxXYhitsCN;
  std::vector<TH2D *> hNHxXYhitsCK;

  // For rechittool z positions. The 0 and 1 are for -ve and +ve, respectively.
  std::vector<TGraph *> grXYhitsF0;
  std::vector<TGraph *> grXYhitsCN0;
  std::vector<TGraph *> grXYhitsCK0;
  int ixyF0[50], ixyCN0[50], ixyCK0[50];

  std::vector<TGraph *> grXYhitsF1;
  std::vector<TGraph *> grXYhitsCN1;
  std::vector<TGraph *> grXYhitsCK1;
  int ixyF1[50], ixyCN1[50], ixyCK1[50];
  /////////////////////////////////

  // For detid zside. The 0 and 1 are for -ve and +ve, respectively.
  std::vector<TGraph *> gXYhitsF0;
  std::vector<TGraph *> gXYhitsCN0;
  std::vector<TGraph *> gXYhitsCK0;
  int ixydF0[50], ixydCN0[50], ixydCK0[50];

  std::vector<TGraph *> gXYhitsF1;
  std::vector<TGraph *> gXYhitsCN1;
  std::vector<TGraph *> gXYhitsCK1;
  int ixydF1[50], ixydCN1[50], ixydCK1[50];
  //////////////////////////////////////////

  std::vector<TH1D *> hELCSMaxF;
  std::vector<TH1D *> hELCSMaxCN;
  std::vector<TH1D *> hELCSMaxCK;

  std::vector<TH1D *> hHxELCSMaxF;
  std::vector<TH1D *> hHxELCSMaxCN;
  std::vector<TH1D *> hHxELCSMaxCK;
  std::vector<TH1D *> hNHxELCSMaxF;
  std::vector<TH1D *> hNHxELCSMaxCN;
  std::vector<TH1D *> hNHxELCSMaxCK;

  TH2D *hXYLowELosshitsF;
  TH2D *hXYLowELosshitsCN;
  TH2D *hXYLowELosshitsCK;
  TH2D *hXYmissedhits;
  TH2D *hYZLowELosshitsF;
  TH2D *hYZLowELosshitsCN;
  TH2D *hYZLowELosshitsCK;
  TH2D *hYZLLowELosshitsHEFCN;
  TH2D *hYZmissedhits;

  TH1D *hXLowELosshitsHEFCN;
  TH1D *hYLowELosshitsHEFCN;
  TH1D *hZLowELosshitsHEFCN;

  TH2D *hYZhitsEE;
  TH2D *hYZhitsHEF;
  TH2D *hYZhitsHEB;

  TH2D *hYZhitsEEF;
  TH2D *hYZhitsEECN;
  TH2D *hYZhitsEECK;

  TH2D *hYZhitsHEFF;
  TH2D *hYZhitsHEFCN;
  TH2D *hYZhitsHEFCK;

  TH2D *hRHTXYhits;
  TH2D *hRHTYZhitsEE;
  TH2D *hRHTYZhitsHEF;
  TH2D *hRHTYZhitsHEB;
  TH2D *hRHTYZhitsEEF;
  TH2D *hRHTYZhitsEECN;
  TH2D *hRHTYZhitsEECK;
  TH2D *hRHTYZhitsHEFF;
  TH2D *hRHTYZhitsHEFCN;
  TH2D *hRHTYZhitsHEFCK;

  TH2D *hRHTRZhitsEE;
  TH2D *hRHTRZhitsHEF;
  TH2D *hRHTRZhitsHEB;
  TH2D *hRHTRZhitsEEF;
  TH2D *hRHTRZhitsEECN;
  TH2D *hRHTRZhitsEECK;
  TH2D *hRHTRZhitsHEFF;
  TH2D *hRHTRZhitsHEFCN;
  TH2D *hRHTRZhitsHEFCK;

  TH2D *hRHTGlbRZhitsF;
  TH2D *hRHTGlbRZhitsCN;
  TH2D *hRHTGlbRZhitsCK;
  TH2D *hRHTGlbRZhitsSci;

  TH1D *hDiffX;
  TH1D *hDiffY;
  TH1D *hDiffZ;

  TH1D *hCellThickness;

  hgcal::RecHitTools rhtools_;
  std::vector<waferinfo> winfo;
  int evt;
};

//
// constructors and destructor
//
HGCalCellHitSum::HGCalCellHitSum(const edm::ParameterSet &iConfig)
    : tSimTrackContainer(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simtrack"))),
      tSimCaloHitContainer(consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("simhits"))),
      name_(iConfig.getParameter<std::string>("detector")),
      geometryFileName_(iConfig.getParameter<edm::FileInPath>("geometryFileName")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_})),
      caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      evt(0) {
  //now do what ever initialization is needed
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  hCharge = fs->make<TH1D>("charge", "Charges", 200, -20, 20);
  hChargeLowELoss = fs->make<TH1D>("charge LowELoss", "Charges LowELoss", 200, -20, 20);

  hPDG = fs->make<TH1D>("hPDG", "hPDG", 10000, -5000, 5000);
  hPDGLowELoss = fs->make<TH1D>("hPDGLowELoss", "hPDGLowELoss", 10000, -5000, 5000);

  hPt = fs->make<TH1D>("hPt", "hPt", 1000, 0., 1000.);
  hPtNoGen = fs->make<TH1D>("hPtNoGen", "hPtNoGen", 1000, 0., 1000.);
  hPtLowELoss = fs->make<TH1D>("hPtLowELoss", "hPtLowELoss", 1000, 0., 1000.);

  hEta = fs->make<TH1D>("hEta", "hEta", 100, -5., 5.);
  hEtaCell = fs->make<TH1D>("hEtaCell", "hEtaCell", 100, -5., 5.);
  hEtaLowELoss = fs->make<TH1D>("hEtaLowELoss", "hEtaLowELoss", 100, -5., 5.);

  hPhi = fs->make<TH1D>("hPhi", "hPhi", 100, -5., 5.);
  hPhiCell = fs->make<TH1D>("hPhiCell", "hPhiCell", 100, -5., 5.);
  hPhiLowELoss = fs->make<TH1D>("hPhiLowELoss", "hPhiLowELoss", 100, -5., 5.);

  hELossEE = fs->make<TH1D>("hELossEE", "hELossEE", 1000, 0., 1000.);
  hELossEEF = fs->make<TH1D>("hELossEEF", "hELossEEF", 1000, 0., 1000.);
  hELossEECN = fs->make<TH1D>("hELossEECN", "hELossEECN", 1000, 0., 1000.);
  hELossEECK = fs->make<TH1D>("hELossEECK", "hELossEECK", 1000, 0., 1000.);

  hELossHEF = fs->make<TH1D>("hELossHEF", "hELossHEF", 1000, 0., 1000.);
  hELossHEFF = fs->make<TH1D>("hELossHEFF", "hELossHEFF", 1000, 0., 1000.);
  hELossHEFCN = fs->make<TH1D>("hELossHEFCN", "hELossHEFCN", 1000, 0., 1000.);
  hELossHEFCK = fs->make<TH1D>("hELossHEFCK", "hELossHEFCK", 1000, 0., 1000.);

  hELossHEB = fs->make<TH1D>("hELossHEB", "hELossHEB", 1000, 0., 1000.);

  hELossCSinBunchEE = fs->make<TH1D>("hELossCSinBunchEE", "hELossCSinBunchEE", 1000, 0., 1000.);
  hELossCSinBunchEEF = fs->make<TH1D>("hELossCSinBunchEEF", "hELossCSinBunchEEF", 1000, 0., 1000.);
  hELossCSinBunchEECN = fs->make<TH1D>("hELossCSinBunchEECN", "hELossCSinBunchEECN", 1000, 0., 1000.);
  hELossCSinBunchEECK = fs->make<TH1D>("hELossCSinBunchEECK", "hELossCSinBunchEECK", 1000, 0., 1000.);
  hELossCSinBunchHEF = fs->make<TH1D>("hELossCSinBunchHEF", "hELossCSinBunchHEF", 1000, 0., 1000.);
  hELossCSinBunchHEFF = fs->make<TH1D>("hELossCSinBunchHEFF", "hELossCSinBunchHEFF", 1000, 0., 1000.);
  hELossCSinBunchHEFCN = fs->make<TH1D>("hELossCSinBunchHEFCN", "hELossCSinBunchHEFCN", 1000, 0., 1000.);
  hELossCSinBunchHEFCK = fs->make<TH1D>("hELossCSinBunchHEFCK", "hELossCSinBunchHEFCK", 1000, 0., 1000.);
  hELossCSinBunchHEFCNFiltered =
      fs->make<TH1D>("hELossCSinBunchHEFCNFiltered", "hELossCSinBunchHEFCNFiltered", 1000, 0., 1000.);
  hELossCSinBunchHEFCNNoise = fs->make<TH1D>("hELossCSinBunchHEFCNNoise", "hELossCSinBunchHEFCNNoise", 1000, 0., 1000.);

  hELossCSmissedEE = fs->make<TH1D>("hELossCSmissedEE", "hELossCSmissedEE", 1000, 0., 1000.);
  hELossCSmissedEEF = fs->make<TH1D>("hELossCSmissedEEF", "hELossCSmissedEEF", 1000, 0., 1000.);
  hELossCSmissedEECN = fs->make<TH1D>("hELossCSmissedEECN", "hELossCSmissedEECN", 1000, 0., 1000.);
  hELossCSmissedEECK = fs->make<TH1D>("hELossCSmissedEECK", "hELossCSmissedEECK", 1000, 0., 1000.);
  hELossCSmissedHEF = fs->make<TH1D>("hELossCSmissedHEF", "hELossCSmissedHEF", 1000, 0., 1000.);
  hELossCSmissedHEFF = fs->make<TH1D>("hELossCSmissedHEFF", "hELossCSmissedHEFF", 1000, 0., 1000.);
  hELossCSmissedHEFCN = fs->make<TH1D>("hELossCSmissedHEFCN", "hELossCSmissedHEFCN", 1000, 0., 1000.);
  hELossCSmissedHEFCK = fs->make<TH1D>("hELossCSmissedHEFCK", "hELossCSmissedHEFCK", 1000, 0., 1000.);

  hELossCSMaxEE = fs->make<TH1D>("hELossCSMaxEE", "hELossCSMaxEE", 1000, 0., 1000.);
  hELossCSMaxEEF = fs->make<TH1D>("hELossCSMaxEEF", "hELossCSMaxEEF", 1000, 0., 1000.);
  hELossCSMaxEECN = fs->make<TH1D>("hELossCSMaxEECN", "hELossCSMaxEECN", 1000, 0., 1000.);
  hELossCSMaxEECK = fs->make<TH1D>("hELossCSMaxEECK", "hELossCSMaxEECK", 1000, 0., 1000.);
  hELossCSMaxHEF = fs->make<TH1D>("hELossCSMaxHEF", "hELossCSMaxHEF", 1000, 0., 1000.);
  hELossCSMaxHEFF = fs->make<TH1D>("hELossCSMaxHEFF", "hELossCSMaxHEFF", 1000, 0., 1000.);
  hELossCSMaxHEFCN = fs->make<TH1D>("hELossCSMaxHEFCN", "hELossCSMaxHEFCN", 1000, 0., 1000.);
  hELossCSMaxHEFCK = fs->make<TH1D>("hELossCSMaxHEFCK", "hELossCSMaxHEFCK", 1000, 0., 1000.);

  hHxELossCSMaxF = fs->make<TH1D>("hHxELossCSMaxF", "hHxELossCSMaxF", 1000, 0., 1000.);
  hHxELossCSMaxCN = fs->make<TH1D>("hHxELossCSMaxCN", "hHxELossCSMaxCN", 1000, 0., 1000.);
  hHxELossCSMaxCK = fs->make<TH1D>("hHxELossCSMaxCK", "hHxELossCSMaxCK", 1000, 0., 1000.);

  hNHxELossCSMaxF = fs->make<TH1D>("hNHxELossCSMaxF", "hNHxELossCSMaxF", 1000, 0., 1000.);
  hNHxELossCSMaxCN = fs->make<TH1D>("hNHxELossCSMaxCN", "hNHxELossCSMaxCN", 1000, 0., 1000.);
  hNHxELossCSMaxCK = fs->make<TH1D>("hNHxELossCSMaxCK", "hNHxELossCSMaxCK", 1000, 0., 1000.);

  for (int i = 1; i <= 50; i++) {
    hELCSMaxF.emplace_back(
        fs->make<TH1D>(Form("hELCSMaxF_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hELCSMaxCN.emplace_back(
        fs->make<TH1D>(Form("hELCSMaxCN_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hELCSMaxCK.emplace_back(
        fs->make<TH1D>(Form("hELCSMaxCK_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
  }
  for (int i = 1; i <= 50; i++) {
    hHxELCSMaxF.emplace_back(
        fs->make<TH1D>(Form("hHxELCSMaxF_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hHxELCSMaxCN.emplace_back(
        fs->make<TH1D>(Form("hHxELCSMaxCN_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hHxELCSMaxCK.emplace_back(
        fs->make<TH1D>(Form("hHxELCSMaxCK_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hNHxELCSMaxF.emplace_back(
        fs->make<TH1D>(Form("hNHxELCSMaxF_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hNHxELCSMaxCN.emplace_back(
        fs->make<TH1D>(Form("hNHxELCSMaxCN_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hNHxELCSMaxCK.emplace_back(
        fs->make<TH1D>(Form("hNHxELCSMaxCK_layer_%02d", i), Form("Energy for layer %d", i), 500, 0., 500.));
    hELossDQMEqV.emplace_back(
        fs->make<TH1D>(Form("hELossDQMEqV_layer_%02d", i), Form("hELossDQMEqV_layer_%02d", i), 100, 0, 0.1));
    hELossLayer.emplace_back(fs->make<TH1D>(Form("hELossLayer_%02d", i), Form("hELossLayer_%02d", i), 1000, 0., 1000.));
  }
  for (int i = 1; i <= 50; i++) {
    hXYhits.emplace_back(fs->make<TH2D>(
        Form("hXYhits_layer_%02d", i), Form("Hits in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsF.emplace_back(fs->make<TH2D>(
        Form("hXYhitsF_layer_%02d", i), Form("HitsF in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsCN.emplace_back(fs->make<TH2D>(
        Form("hXYhitsCN_layer_%02d", i), Form("HitsCN in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsCK.emplace_back(fs->make<TH2D>(
        Form("hXYhitsCK_layer_%02d", i), Form("HitsCK in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsB.emplace_back(fs->make<TH2D>(
        Form("hXYhitsB_layer_%02d", i), Form("HitsB in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsLELCN.emplace_back(fs->make<TH2D>(
        Form("hXYhitsLELCN_layer_%02d", i), Form("LELCN in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsHELCN.emplace_back(fs->make<TH2D>(
        Form("hXYhitsHELCN_layer_%02d", i), Form("HELCN in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsLELCK.emplace_back(fs->make<TH2D>(
        Form("hXYhitsLELCK_layer_%02d", i), Form("LELCK in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hXYhitsHELCK.emplace_back(fs->make<TH2D>(
        Form("hXYhitsHELCK_layer_%02d", i), Form("HELCK in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
  }
  for (int i = 1; i <= 50; i++) {
    grXYhitsF0.emplace_back(fs->make<TGraph>(0));
    grXYhitsF0[i - 1]->SetNameTitle(Form("grXYhitsF0_layer_%02d", i), Form("HitsF0 in XY for layer %d", i));
    grXYhitsCN0.emplace_back(fs->make<TGraph>(0));
    grXYhitsCN0[i - 1]->SetNameTitle(Form("grXYhitsCN0_layer_%02d", i), Form("HitsCN0 in XY for layer %d", i));
    grXYhitsCK0.emplace_back(fs->make<TGraph>(0));
    grXYhitsCK0[i - 1]->SetNameTitle(Form("grXYhitsCK0_layer_%02d", i), Form("HitsCK0 in XY for layer %d", i));
    ixyF0[i - 1] = 0;
    ixyCN0[i - 1] = 0;
    ixyCK0[i - 1] = 0;
    grXYhitsF1.emplace_back(fs->make<TGraph>(0));
    grXYhitsF1[i - 1]->SetNameTitle(Form("grXYhitsF1_layer_%02d", i), Form("HitsF1 in XY for layer %d", i));
    grXYhitsCN1.emplace_back(fs->make<TGraph>(0));
    grXYhitsCN1[i - 1]->SetNameTitle(Form("grXYhitsCN1_layer_%02d", i), Form("HitsCN1 in XY for layer %d", i));
    grXYhitsCK1.emplace_back(fs->make<TGraph>(0));
    grXYhitsCK1[i - 1]->SetNameTitle(Form("grXYhitsCK1_layer_%02d", i), Form("HitsCK1 in XY for layer %d", i));
    ixyF1[i - 1] = 0;
    ixyCN1[i - 1] = 0;
    ixyCK1[i - 1] = 0;
    gXYhitsF0.emplace_back(fs->make<TGraph>(0));
    gXYhitsF0[i - 1]->SetNameTitle(Form("gXYhitsF0_layer_%02d", i), Form("HitsF0 in XY for layer %d", i));
    gXYhitsCN0.emplace_back(fs->make<TGraph>(0));
    gXYhitsCN0[i - 1]->SetNameTitle(Form("gXYhitsCN0_layer_%02d", i), Form("HitsCN0 in XY for layer %d", i));
    gXYhitsCK0.emplace_back(fs->make<TGraph>(0));
    gXYhitsCK0[i - 1]->SetNameTitle(Form("gXYhitsCK0_layer_%02d", i), Form("HitsCK0 in XY for layer %d", i));
    ixydF0[i - 1] = 0;
    ixydCN0[i - 1] = 0;
    ixydCK0[i - 1] = 0;
    gXYhitsF1.emplace_back(fs->make<TGraph>(0));
    gXYhitsF1[i - 1]->SetNameTitle(Form("gXYhitsF1_layer_%02d", i), Form("HitsF1 in XY for layer %d", i));
    gXYhitsCN1.emplace_back(fs->make<TGraph>(0));
    gXYhitsCN1[i - 1]->SetNameTitle(Form("gXYhitsCN1_layer_%02d", i), Form("HitsCN1 in XY for layer %d", i));
    gXYhitsCK1.emplace_back(fs->make<TGraph>(0));
    gXYhitsCK1[i - 1]->SetNameTitle(Form("gXYhitsCK1_layer_%02d", i), Form("HitsCK1 in XY for layer %d", i));
    ixydF1[i - 1] = 0;
    ixydCN1[i - 1] = 0;
    ixydCK1[i - 1] = 0;
  }
  for (int i = 1; i <= 50; i++) {
    hNHxXYhitsF.emplace_back(fs->make<TH2D>(
        Form("hNHxXYhitsF_layer_%02d", i), Form("NHx HitsF in XY for layer %d", i), 600, -300., 300., 600, -300., 300.));
    hNHxXYhitsCN.emplace_back(fs->make<TH2D>(Form("hNHxXYhitsCN_layer_%02d", i),
                                             Form("NHx HitsCN in XY for layer %d", i),
                                             600,
                                             -300.,
                                             300.,
                                             600,
                                             -300.,
                                             300.));
    hNHxXYhitsCK.emplace_back(fs->make<TH2D>(Form("hNHxXYhitsCK_layer_%02d", i),
                                             Form("NHx HitsCK in XY for layer %d", i),
                                             600,
                                             -300.,
                                             300.,
                                             600,
                                             -300.,
                                             300.));
  }
  hXYmissedhits = fs->make<TH2D>("hXYmissedhits", "hXYmissedhits", 600, -300., 300., 600, -300., 300.);
  hXYLowELosshitsF = fs->make<TH2D>("hXYLowELosshitsF", "hXYLowELosshitsF", 600, -300., 300., 600, -300., 300.);
  hXYLowELosshitsCN = fs->make<TH2D>("hXYLowELosshitsCN", "hXYLowELosshitsCN", 600, -300., 300., 600, -300., 300.);
  hXYLowELosshitsCK = fs->make<TH2D>("hXYLowELosshitsCK", "hXYLowELosshitsCK", 600, -300., 300., 600, -300., 300.);

  hYZmissedhits = fs->make<TH2D>("hYZmissedhits", "hYZmissedhits", 250, 300., 550., 300, 0., 300.);
  hYZLowELosshitsF = fs->make<TH2D>("hYZLowELosshitsF", "hYZLowELosshitsF", 250, 300., 550., 300, 0., 300.);
  hYZLowELosshitsCN = fs->make<TH2D>("hYZLowELosshitsCN", "hYZLowELosshitsCN", 250, 300., 550., 300, 0., 300.);
  hYZLowELosshitsCK = fs->make<TH2D>("hYZLowELosshitsCK", "hYZLowELosshitsCK", 250, 300., 550., 300, 0., 300.);
  hYZLLowELosshitsHEFCN =
      fs->make<TH2D>("hYZLLowELosshitsHEFCN", "hYZLLowELosshitsHEFCN", 600, -50., 550., 350, -50., 300.);

  hXLowELosshitsHEFCN = fs->make<TH1D>("hXLowELosshitsHEFCN", "hXLowELosshitsHEFCN", 600, -300., 300.);
  hYLowELosshitsHEFCN = fs->make<TH1D>("hYLowELosshitsHEFCN", "hYLowELosshitsHEFCN", 600, -300., 300.);
  hZLowELosshitsHEFCN = fs->make<TH1D>("hZLowELosshitsHEFCN", "hZLowELosshitsHEFCN", 2400, -1200., 1200.);

  hYZhitsEE = fs->make<TH2D>("hYZhitsEE", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hYZhitsHEF = fs->make<TH2D>("hYZhitsHEF", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hYZhitsHEB = fs->make<TH2D>("hYZhitsHEB", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);

  hYZhitsEEF = fs->make<TH2D>("hYZhitsEEF", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hYZhitsEECN = fs->make<TH2D>("hYZhitsEECN", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hYZhitsEECK = fs->make<TH2D>("hYZhitsEECK", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);

  hYZhitsHEFF = fs->make<TH2D>("hYZhitsHEFF", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hYZhitsHEFCN = fs->make<TH2D>("hYZhitsHEFCN", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hYZhitsHEFCK = fs->make<TH2D>("hYZhitsHEFCK", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);

  hRHTXYhits = fs->make<TH2D>("hRHTXYhits", "Hits in XY", 600, -300., 300., 600, -300., 300.);
  hRHTYZhitsEE = fs->make<TH2D>("hRHTYZhitsEE", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsHEF = fs->make<TH2D>("hRHTYZhitsHEF", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsHEB = fs->make<TH2D>("hRHTYZhitsHEB", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsEEF = fs->make<TH2D>("hRHTYZhitsEEF", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsEECN = fs->make<TH2D>("hRHTYZhitsEECN", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsEECK = fs->make<TH2D>("hRHTYZhitsEECK", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsHEFF = fs->make<TH2D>("hRHTYZhitsHEFF", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsHEFCN =
      fs->make<TH2D>("hRHTYZhitsHEFCN", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTYZhitsHEFCK =
      fs->make<TH2D>("hRHTYZhitsHEFCK", "Hits in YZ plane for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);

  hRHTRZhitsEE =
      fs->make<TH2D>("hRHTRZhitsEE", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsHEF =
      fs->make<TH2D>("hRHTRZhitsHEF", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsHEB =
      fs->make<TH2D>("hRHTRZhitsHEB", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsEEF =
      fs->make<TH2D>("hRHTRZhitsEEF", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsEECN =
      fs->make<TH2D>("hRHTRZhitsEECN", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsEECK =
      fs->make<TH2D>("hRHTRZhitsEECK", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsHEFF =
      fs->make<TH2D>("hRHTRZhitsHEFF", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsHEFCN =
      fs->make<TH2D>("hRHTRZhitsHEFCN", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);
  hRHTRZhitsHEFCK =
      fs->make<TH2D>("hRHTRZhitsHEFCK", "Hits for R_{xy} vs z-axis for |X| < 20 cm", 250, 300., 550., 300, 0., 300.);

  hRHTGlbRZhitsF = fs->make<TH2D>("hRHTGlbRZhitsF", "Hits for R_{xy} vs z-axis", 250, 300., 550., 300, 0., 300.);
  hRHTGlbRZhitsCN = fs->make<TH2D>("hRHTGlbRZhitsCN", "Hits for R_{xy} vs z-axis", 250, 300., 550., 300, 0., 300.);
  hRHTGlbRZhitsCK = fs->make<TH2D>("hRHTGlbRZhitsCK", "Hits for R_{xy} vs z-axis", 250, 300., 550., 300, 0., 300.);
  hRHTGlbRZhitsSci = fs->make<TH2D>("hRHTGlbRZhitsSci", "Hits for R_{xy} vs z-axis", 250, 300., 550., 300, 0., 300.);

  hDiffX = fs->make<TH1D>("hDiffX", "Difference of x-position (testHGCalGeometry - RecHitTools)", 200, -20, 20);
  hDiffX->GetXaxis()->SetTitle("x-axis (cm)");
  hDiffY = fs->make<TH1D>("hDiffY", "Difference of y-position (testHGCalGeometry - RecHitTools)", 200, -20, 20);
  hDiffY->GetXaxis()->SetTitle("y-axis (cm)");
  hDiffZ = fs->make<TH1D>("hDiffZ", "Difference of z-position (testHGCalGeometry - RecHitTools)", 200, -20, 20);
  hDiffZ->GetXaxis()->SetTitle("z-axis (cm)");

  hCellThickness = fs->make<TH1D>("hCellThickness", "Cell Thickness", 500, 0, 500);
  hDiffZ->GetXaxis()->SetTitle("thickness (#mum)");

  evt = 0;
  winfo.clear();
}

//
// member functions
//

// ------------ method called for each event  ------------
void HGCalCellHitSum::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (evt == 0) {
    std::string fileName = geometryFileName_.fullPath();
    std::ifstream fin(fileName);
    std::string s;
    waferinfo wafer;
    //std::cout << "evt : " << evt << std::endl;
    while (std::getline(fin, s)) {
      //std::cout << "line " << s.data() << std::endl;
      sscanf(s.c_str(), "%d,%d,%d,%d", &wafer.layer, &wafer.u, &wafer.v, &wafer.type);
      //printf("%d | %d | %d | %d\n",wafer.layer,wafer.u,wafer.v,wafer.type);
      //wafer.layer =
      winfo.push_back(wafer);
    }
    fin.close();
  }
  evt++;

  const edm::Handle<edm::SimTrackContainer> &simtrack = iEvent.getHandle(tSimTrackContainer);
  int itrk = 0;
  //double muonpt = 0.0;
  edm::SimTrackContainer::const_iterator itTrack;
  for (itTrack = simtrack->begin(); itTrack != simtrack->end(); ++itTrack) {
    // int charge = itTrack->charge();
    int charge = itTrack->charge();
    hCharge->Fill(charge);
    if (!itTrack->noGenpart()) {
      hPt->Fill(itTrack->momentum().pt());
      hEta->Fill(itTrack->momentum().eta());
      hPhi->Fill(itTrack->momentum().phi());
    }
    hPDG->Fill(itTrack->type());

    if (itTrack->noGenpart())
      hPtNoGen->Fill(itTrack->momentum().pt());

    itrk++;
  }

  const CaloGeometry &geomCalo = iSetup.getData(caloGeomToken_);
  rhtools_.setGeometry(geomCalo);

  const HGCalGeometry *geom = &iSetup.getData(geomToken_);

  std::map<uint32_t, std::pair<hitsinfo, energysum> > map_hits;
  map_hits.clear();
  unsigned int nofSiHits = 0;
  const edm::Handle<edm::PCaloHitContainer> &simhit = iEvent.getHandle(tSimCaloHitContainer);
  for (edm::PCaloHitContainer::const_iterator itHit = simhit->begin(); itHit != simhit->end(); ++itHit) {
    if ((name_ == "HGCalEESensitive") || (name_ == "HGCalHESiliconSensitive")) {
      HGCSiliconDetId id(itHit->id());

      if (name_ == "HGCalEESensitive") {
        hELossEE->Fill(convertGeVToKeV(itHit->energy()));
        if (id.type() == HGCSiliconDetId::HGCalFine)
          hELossEEF->Fill(convertGeVToKeV(itHit->energy()));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin)
          hELossEECN->Fill(convertGeVToKeV(itHit->energy()));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick)
          hELossEECK->Fill(convertGeVToKeV(itHit->energy()));  //in keV
      }

      if (name_ == "HGCalHESiliconSensitive") {
        hELossHEF->Fill(convertGeVToKeV(itHit->energy()));
        if (id.type() == HGCSiliconDetId::HGCalFine)
          hELossHEFF->Fill(convertGeVToKeV(itHit->energy()));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin)
          hELossHEFCN->Fill(convertGeVToKeV(itHit->energy()));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick)
          hELossHEFCK->Fill(convertGeVToKeV(itHit->energy()));  //in keV
      }
    }

    if (name_ == "HGCalHEScintillatorSensitive")
      hELossHEB->Fill(convertGeVToKeV(itHit->energy()));

    DetId id1 = static_cast<DetId>(itHit->id());
    GlobalPoint global2 = rhtools_.getPosition(id1);
    double RXY = TMath::Sqrt(global2.x() * global2.x() + global2.y() * global2.y());

    // std::cout << "DetId (" << det << ": position ("<< global2.x() << ", " << global2.y() << ", " << global2.z()
    // 	      << "), Si thickness "<< rhtools_.getSiThickness(id1)
    // 	      << ", IsSi "<< rhtools_.isSilicon(id1)
    // 	      << ", IsSci "<< rhtools_.isScintillator(id1)
    // 	      << ", Layer1 "<< rhtools_.getLayer(id1)
    // 	      << ", Layer2 "<< rhtools_.getLayerWithOffset(id1)
    // 	      << ", lastLayerEE  "<< rhtools_.lastLayerEE()
    // 	      << ", lastLayerFH  "<< rhtools_.lastLayerFH()
    // 	      << ", firstLayerBH  "<< rhtools_.firstLayerBH()
    // 	      << ", lastLayerBH  "<< rhtools_.lastLayerBH()
    // 	      << ", lastLayer  "<< rhtools_.lastLayer()
    // 	      << std::endl;

    //if ((rhtools_.isSilicon(id1) or rhtools_.isScintillator(id1)) and TMath::Abs(global2.x())<20.0){
    if ((rhtools_.isSilicon(id1)) || (rhtools_.isScintillator(id1))) {
      if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 120., 1.e-7))
        hRHTGlbRZhitsF->Fill(TMath::Abs(global2.z()), RXY);
      else if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 200., 1.e-7))
        hRHTGlbRZhitsCN->Fill(TMath::Abs(global2.z()), RXY);
      else if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 300., 1.e-7))
        hRHTGlbRZhitsCK->Fill(TMath::Abs(global2.z()), RXY);
      else
        hRHTGlbRZhitsSci->Fill(TMath::Abs(global2.z()), RXY);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    if ((rhtools_.isSilicon(id1)) || (rhtools_.isScintillator(id1))) {
      uint32_t id_ = itHit->id();

      energysum esum;
      hitsinfo hinfo;

      if (map_hits.count(id_) != 0) {
        hinfo = map_hits[id_].first;
        esum = map_hits[id_].second;
      } else {
        hinfo.hitid = nofSiHits;
        hinfo.x = global2.x();
        hinfo.y = global2.y();
        hinfo.z = global2.z();
        hinfo.layer = rhtools_.getLayerWithOffset(id1);
        hinfo.phi = rhtools_.getPhi(id1);
        hinfo.eta = rhtools_.getEta(id1);
        for (itTrack = simtrack->begin(); itTrack != simtrack->end(); ++itTrack) {
          if (itTrack->trackId() == UInt_t(itHit->geantTrackId())) {
            hinfo.trkpt = itTrack->momentum().pt();
            hinfo.trketa = itTrack->momentum().eta();
            hinfo.trkphi = itTrack->momentum().phi();
            hinfo.charge = itTrack->charge();
            hinfo.pdg = itTrack->type();
          }
        }
      }
      esum.etotal += itHit->energy();
      hinfo.nhits++;

      HepGeom::Point3D<float> gcoord = HepGeom::Point3D<float>(global2.x(), global2.y(), global2.z());
      double tof = (gcoord.mag() * CLHEP::cm) / CLHEP::c_light;
      double time = itHit->time();
      time -= tof;

      for (unsigned int k = 0; k < 2; ++k) {
        if (time > 0 && time < 25.)
          esum.eTime[k] += itHit->energy();
        else {
          esum.eTime[k + 2] += itHit->energy();
        }
      }

      map_hits[id_] = std::pair<hitsinfo, energysum>(hinfo, esum);
      nofSiHits++;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    hEtaCell->Fill(rhtools_.getEta(id1));
    hPhiCell->Fill(rhtools_.getPhi(id1));
    ///////////////////////////////////////////////////////////////////////////////////////////////

    GlobalPoint global1 = geom->getPosition(id1);

    if (geom->topology().valid(id1)) {
      //std::cout << "DetId (" << det << ": position ("<< global1.x() << ", " << global1.y() << ", " << global1.z() << ") " << std::endl;

      //hYZhits->Fill(global1.z(),global1.y());
      if (TMath::Abs(global1.x()) < 20.0) {
        if ((name_ == "HGCalEESensitive") || (name_ == "HGCalHESiliconSensitive")) {
          HGCSiliconDetId id(itHit->id());

          if (name_ == "HGCalEESensitive") {
            hYZhitsEE->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
            if (id.type() == HGCSiliconDetId::HGCalFine)
              hYZhitsEEF->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
            if (id.type() == HGCSiliconDetId::HGCalCoarseThin)
              hYZhitsEECN->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
            if (id.type() == HGCSiliconDetId::HGCalCoarseThick)
              hYZhitsEECK->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
          }

          if (name_ == "HGCalHESiliconSensitive") {
            hYZhitsHEF->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
            if (id.type() == HGCSiliconDetId::HGCalFine)
              hYZhitsHEFF->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
            if (id.type() == HGCSiliconDetId::HGCalCoarseThin)
              hYZhitsHEFCN->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
            if (id.type() == HGCSiliconDetId::HGCalCoarseThick)
              hYZhitsHEFCK->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
          }
        }

        if (name_ == "HGCalHEScintillatorSensitive")
          hYZhitsHEB->Fill(TMath::Abs(global1.z()), TMath::Abs(global1.y()));
      }

      /// Using rechit tools
      //===============================================================================
      if (TMath::Abs(global2.x()) < 20.0) {
        if (rhtools_.isSilicon(id1)) {
          if (rhtools_.getLayerWithOffset(id1) <= rhtools_.lastLayerEE()) {
            hRHTYZhitsEE->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
            hRHTRZhitsEE->Fill(TMath::Abs(global2.z()), RXY);

            if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 120., 1.e-7)) {
              hRHTYZhitsEEF->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
              hRHTRZhitsEEF->Fill(TMath::Abs(global2.z()), RXY);
            }
            if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 200., 1.e-7)) {
              hRHTYZhitsEECN->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
              hRHTRZhitsEECN->Fill(TMath::Abs(global2.z()), RXY);
            }
            if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 300., 1.e-7)) {
              hRHTYZhitsEECK->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
              hRHTRZhitsEECK->Fill(TMath::Abs(global2.z()), RXY);
            }

          } else {
            hRHTYZhitsHEF->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
            hRHTRZhitsHEF->Fill(TMath::Abs(global2.z()), RXY);

            if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 120., 1.e-7)) {
              hRHTYZhitsHEFF->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
              hRHTRZhitsHEFF->Fill(TMath::Abs(global2.z()), RXY);
            }
            if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 200., 1.e-7)) {
              hRHTYZhitsHEFCN->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
              hRHTRZhitsHEFCN->Fill(TMath::Abs(global2.z()), RXY);
            }
            if (TMath::AreEqualAbs(rhtools_.getSiThickness(id1), 300., 1.e-7)) {
              hRHTYZhitsHEFCK->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
              hRHTRZhitsHEFCK->Fill(TMath::Abs(global2.z()), RXY);
            }
          }

        }  //is Si

        if (rhtools_.isScintillator(id1)) {
          hRHTYZhitsHEB->Fill(TMath::Abs(global2.z()), TMath::Abs(global2.y()));
          hRHTRZhitsHEB->Fill(TMath::Abs(global2.z()), RXY);
        }
      }
      //===============================================================================
      hRHTXYhits->Fill(global2.x(), global2.y());

      hXYhits[rhtools_.getLayerWithOffset(id1) - 1]->Fill(global2.x(), global2.y());

      if (rhtools_.isSilicon(id1)) {
        HGCSiliconDetId id(itHit->id());
        HGCalDetId hid(itHit->id());
        int il = rhtools_.getLayerWithOffset(id1) - 1;
        if (id.type() == HGCSiliconDetId::HGCalFine) {
          hXYhitsF[il]->Fill(global2.x(), global2.y());

          if (global2.z() < 0.0)
            grXYhitsF0[il]->SetPoint(ixyF0[il]++, global2.x(), global2.y());
          else
            grXYhitsF1[il]->SetPoint(ixyF1[il]++, global2.x(), global2.y());
          if (id.zside() == -1)
            gXYhitsF0[il]->SetPoint(ixydF0[il]++, global2.x(), global2.y());
          else
            gXYhitsF1[il]->SetPoint(ixydF1[il]++, global2.x(), global2.y());
        }
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
          hXYhitsCN[il]->Fill(global2.x(), global2.y());

          if (global2.z() < 0.0)
            grXYhitsCN0[il]->SetPoint(ixyCN0[il]++, global2.x(), global2.y());
          else
            grXYhitsCN1[il]->SetPoint(ixyCN1[il]++, global2.x(), global2.y());
          if (id.zside() == -1)
            gXYhitsCN0[il]->SetPoint(ixydCN0[il]++, global2.x(), global2.y());
          else
            gXYhitsCN1[il]->SetPoint(ixydCN1[il]++, global2.x(), global2.y());
        }
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {  //case 2 :
          hXYhitsCK[il]->Fill(global2.x(), global2.y());

          if (global2.z() < 0.0)
            grXYhitsCK0[il]->SetPoint(ixyCK0[il]++, global2.x(), global2.y());
          else
            grXYhitsCK1[il]->SetPoint(ixyCK1[il]++, global2.x(), global2.y());
          if (id.zside() == -1)
            gXYhitsCK0[il]->SetPoint(ixydCK0[il]++, global2.x(), global2.y());
          else
            gXYhitsCK1[il]->SetPoint(ixydCK1[il]++, global2.x(), global2.y());
        }
      } else if (rhtools_.isScintillator(id1)) {
        hXYhitsB[rhtools_.getLayerWithOffset(id1)]->Fill(global2.x(), global2.y());
      }
    }

    hDiffX->Fill(global1.x() - global2.x());
    hDiffY->Fill(global1.y() - global2.y());
    hDiffZ->Fill(global1.z() - global2.z());
  }
  //std::cout << "simhit size : " << simhit->size() << ", nof hits in Si : " << nofSiHits << ", map size : " << map_hits.size() << std::endl;

  bool isPWafer = false;
  bool isFWafer = false;

  std::map<uint32_t, std::pair<hitsinfo, energysum> >::iterator itr;
  for (itr = map_hits.begin(); itr != map_hits.end(); ++itr) {
    //uint32_t id_ = (*itr).first;
    hitsinfo hinfo = (*itr).second.first;
    energysum esum = (*itr).second.second;
    hELossDQMEqV[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));

    // printf("\tCellSummed : Det : %s, first hit : %d, nhits : %u, id : %u, Edep : %5.2lf (keV), (x,y,z) : (%5.2lf,%5.2lf,%5.2lf)\n",
    //  	   name_.c_str(), hinfo.hitid, hinfo.nhits, (*itr).first, convertGeVToKeV(esum.eTime[0]), hinfo.x, hinfo.y, hinfo.z);

    HGCSiliconDetId id((*itr).first);

    // DetId id1 = static_cast<DetId>((*itr).first);

    // isPWafer = false;
    // isFWafer = false;
    // if(rhtools_.isSilicon(id1)){
    //   for(unsigned int iw = 0 ; iw < winfo.size() ; iw++){
    // 	if(hinfo.layer == winfo[iw].layer and rhtools_.getWafer(id1).first == winfo[iw].u and rhtools_.getWafer(id1).second == winfo[iw].v){
    // 	  if(winfo[iw].type == 0)
    // 	    isPWafer = true;
    // 	  if(winfo[iw].type == 1)
    // 	    isFWafer = true;
    // 	}
    //   }
    // }

    if (!TMath::AreEqualAbs(convertGeVToKeV(esum.eTime[0]), 0.0, 1.e-5)) {
      if (name_ == "HGCalEESensitive") {
        hELossCSinBunchEE->Fill(convertGeVToKeV(esum.eTime[0]));
        if (id.type() == HGCSiliconDetId::HGCalFine) {
          hELossCSinBunchEEF->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        }
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
          hELossCSinBunchEECN->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        }
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {
          hELossCSinBunchEECK->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        }
      }

      if (name_ == "HGCalHESiliconSensitive") {
        hELossCSinBunchHEF->Fill(convertGeVToKeV(esum.eTime[0]));
        if (id.type() == HGCSiliconDetId::HGCalFine) {
          hELossCSinBunchHEFF->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
          if (convertGeVToKeV(esum.eTime[0]) < 35.) {
            hXYLowELosshitsF->Fill(hinfo.x, hinfo.y);
            hYZLowELosshitsF->Fill(TMath::Abs(hinfo.z), TMath::Sqrt(hinfo.x * hinfo.x + hinfo.y * hinfo.y));
          }
        }
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
          hELossCSinBunchHEFCN->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
          if (TMath::Sqrt(hinfo.x * hinfo.x + hinfo.y * hinfo.y) > 45.0 and
              TMath::Sqrt(hinfo.x * hinfo.x + hinfo.y * hinfo.y) < 60.0 and hinfo.layer >= 38)
            hELossCSinBunchHEFCNNoise->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
          else
            hELossCSinBunchHEFCNFiltered->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
          if (convertGeVToKeV(esum.eTime[0]) < 35.) {
            hPtLowELoss->Fill(hinfo.trkpt);
            hEtaLowELoss->Fill(hinfo.trketa);
            hPhiLowELoss->Fill(hinfo.trkphi);
            hChargeLowELoss->Fill(hinfo.charge);
            hPDGLowELoss->Fill(hinfo.pdg);
            hXYLowELosshitsCN->Fill(hinfo.x, hinfo.y);
            hYZLowELosshitsCN->Fill(TMath::Abs(hinfo.z), TMath::Sqrt(hinfo.x * hinfo.x + hinfo.y * hinfo.y));
            hYZLLowELosshitsHEFCN->Fill(TMath::Abs(hinfo.z), TMath::Sqrt(hinfo.x * hinfo.x + hinfo.y * hinfo.y));
            hXLowELosshitsHEFCN->Fill(hinfo.x);
            hYLowELosshitsHEFCN->Fill(hinfo.y);
            if (TMath::Abs(hinfo.x) < 20.0 && TMath::Abs(hinfo.y) < 20.0)
              hZLowELosshitsHEFCN->Fill(hinfo.z);
          }
        }
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {
          hELossCSinBunchHEFCK->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
          if (convertGeVToKeV(esum.eTime[0]) < 10.) {
            hXYLowELosshitsCK->Fill(hinfo.x, hinfo.y);
            hYZLowELosshitsCK->Fill(TMath::Abs(hinfo.z), TMath::Sqrt(hinfo.x * hinfo.x + hinfo.y * hinfo.y));
          }
        }
      }
    }

    if (!TMath::AreEqualAbs(convertGeVToKeV(esum.eTime[2]), 0.0, 1.e-5)) {
      if (name_ == "HGCalEESensitive") {
        hELossCSmissedEE->Fill(convertGeVToKeV(esum.eTime[2]));
        if (id.type() == HGCSiliconDetId::HGCalFine)
          hELossCSmissedEEF->Fill(convertGeVToKeV(esum.eTime[2]));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin)
          hELossCSmissedEECN->Fill(convertGeVToKeV(esum.eTime[2]));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick)
          hELossCSmissedEECK->Fill(convertGeVToKeV(esum.eTime[2]));  //in keV
        hXYmissedhits->Fill(hinfo.x, hinfo.y);
        hYZmissedhits->Fill(TMath::Abs(hinfo.z), TMath::Abs(hinfo.y));
      }

      if (name_ == "HGCalHESiliconSensitive") {
        hELossCSmissedHEF->Fill(convertGeVToKeV(esum.eTime[2]));
        if (id.type() == HGCSiliconDetId::HGCalFine)
          hELossCSmissedHEFF->Fill(convertGeVToKeV(esum.eTime[2]));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThin)
          hELossCSmissedHEFCN->Fill(convertGeVToKeV(esum.eTime[2]));  //in keV
        if (id.type() == HGCSiliconDetId::HGCalCoarseThick)
          hELossCSmissedHEFCK->Fill(convertGeVToKeV(esum.eTime[2]));  //in keV
        hXYmissedhits->Fill(hinfo.x, hinfo.y);
        hYZmissedhits->Fill(TMath::Abs(hinfo.z), TMath::Abs(hinfo.y));
      }
    }
  }

  std::vector<uint32_t> cellMaxEdep;
  cellMaxEdep.clear();
  for (int il = 1; il <= 50; il++) {
    double energy = 0.;
    uint32_t maxid = 0;
    double maxEsum = 0.0;
    for (itr = map_hits.begin(); itr != map_hits.end(); ++itr) {
      //uint32_t id_ = (*itr).first;
      hitsinfo hinfo = (*itr).second.first;
      energysum esum = (*itr).second.second;
      // printf("\tDet : %s, first hit : %d, nhits : %u, id : %u, Edep : %5.2lf (keV), (x,y,z) : (%lf,%lf,%lf)\n",
      // 	   name_.c_str(), hinfo.hitid, hinfo.nhits, (*itr).first, convertGeVToKeV(esum.etotal), hinfo.x, hinfo.y, hinfo.z);

      if (hinfo.layer == il and hinfo.z > 0.) {
        energy += esum.eTime[0];

        if (esum.eTime[0] > maxEsum) {
          maxEsum = esum.eTime[0];
          maxid = (*itr).first;
        }

      }  //match layer and z-direction

    }  //map loop

    if (convertGeVToKeV(maxEsum) > 0.)
      cellMaxEdep.push_back(maxid);
    if (convertGeVToKeV(energy) > 0.)
      hELossLayer[il - 1]->Fill(convertGeVToKeV(energy));  //in keV
  }

  // waferinfo wafer;
  // winfo.push_back(wafer);
  //std::cout<<"size :: "<<winfo.size()<<std::endl;

  for (unsigned int ic = 0; ic < cellMaxEdep.size(); ic++) {
    uint32_t id_ = cellMaxEdep[ic];
    energysum esum = map_hits[id_].second;
    hitsinfo hinfo = map_hits[id_].first;
    DetId id1 = static_cast<DetId>(id_);

    if (!rhtools_.isSilicon(id1))
      continue;

    HGCSiliconDetId id(id_);
    HGCalDetId hid(id);

    isPWafer = false;
    isFWafer = false;
    for (unsigned int iw = 0; iw < winfo.size(); iw++) {
      if (hinfo.layer == winfo[iw].layer and rhtools_.getWafer(id1).first == winfo[iw].u and
          rhtools_.getWafer(id1).second == winfo[iw].v) {
        if (winfo[iw].type == 0)
          isPWafer = true;
        if (winfo[iw].type == 1)
          isFWafer = true;
      }
    }

    // printf("\tDet : %s, wafertype : %d, layer : %d, (u,v) : (%d,%d), ishalf : %d, first hit : %d, nhits : %u, Edep : %5.2lf (keV), (x,y,z) : (%lf,%lf,%lf)\n",
    // 	   name_.c_str(), hid.waferType(), hinfo.layer, rhtools_.getWafer(id1).first, rhtools_.getWafer(id1).second, rhtools_.isHalfCell(id1), hinfo.hitid, hinfo.nhits, convertGeVToKeV(esum.etotal), hinfo.x, hinfo.y, hinfo.z);

    // printf("\tDet : %s, wafertype : %d, layer : %d, (u,v) : (%d,%d), isPWafer : %d, isFWafer : %d, (x,y,z) : (%lf,%lf,%lf)\n",
    // 	   name_.c_str(), hid.waferType(), hinfo.layer, rhtools_.getWafer(id1).first, rhtools_.getWafer(id1).second, isPWafer, isFWafer, hinfo.x, hinfo.y, hinfo.z);

    //for

    if (name_ == "HGCalEESensitive") {
      hCellThickness->Fill(rhtools_.getSiThickness(id1));
      hELossCSMaxEE->Fill(convertGeVToKeV(esum.eTime[0]));
      if (id.type() == HGCSiliconDetId::HGCalFine) {
        hELossCSMaxEEF->Fill(convertGeVToKeV(esum.eTime[0]));              //in keV
        hELCSMaxF[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        if (isPWafer) {
          hNHxELossCSMaxF->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxELCSMaxF[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxXYhitsF[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        }
        if (isFWafer) {
          hHxELossCSMaxF->Fill(convertGeVToKeV(esum.eTime[0]));
          hHxELCSMaxF[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
        }
      }
      if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
        hELossCSMaxEECN->Fill(convertGeVToKeV(esum.eTime[0]));              //in keV
        hELCSMaxCN[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        if (isPWafer) {
          hNHxELossCSMaxCN->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxELCSMaxCN[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxXYhitsCN[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        }
        if (isFWafer) {
          hHxELossCSMaxCN->Fill(convertGeVToKeV(esum.eTime[0]));
          hHxELCSMaxCN[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
        }
      }
      if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {
        hELossCSMaxEECK->Fill(convertGeVToKeV(esum.eTime[0]));              //in keV
        hELCSMaxCK[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        if (isPWafer) {
          hNHxELossCSMaxCK->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxELCSMaxCK[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxXYhitsCK[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        }
        if (isFWafer) {
          hHxELossCSMaxCK->Fill(convertGeVToKeV(esum.eTime[0]));
          hHxELCSMaxCK[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
        }
      }
    }

    if (name_ == "HGCalHESiliconSensitive") {
      hCellThickness->Fill(rhtools_.getSiThickness(id1));
      hELossCSMaxHEF->Fill(convertGeVToKeV(esum.eTime[0]));
      if (id.type() == HGCSiliconDetId::HGCalFine) {
        hELossCSMaxHEFF->Fill(convertGeVToKeV(esum.eTime[0]));             //in keV
        hELCSMaxF[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        if (isPWafer) {
          hNHxELossCSMaxF->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxELCSMaxF[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxXYhitsF[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        }
        if (isFWafer) {
          hHxELossCSMaxF->Fill(convertGeVToKeV(esum.eTime[0]));
          hHxELCSMaxF[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
        }
      }
      if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
        hELossCSMaxHEFCN->Fill(convertGeVToKeV(esum.eTime[0]));             //in keV
        hELCSMaxCN[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        if (convertGeVToKeV(esum.eTime[0]) < 30. and convertGeVToKeV(esum.eTime[0]) > 10.)
          hXYhitsLELCN[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        else
          hXYhitsHELCN[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        if (isPWafer) {
          hNHxELossCSMaxCN->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxELCSMaxCN[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxXYhitsCN[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        }
        if (isFWafer) {
          hHxELossCSMaxCN->Fill(convertGeVToKeV(esum.eTime[0]));
          hHxELCSMaxCN[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
        }
      }
      if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {
        hELossCSMaxHEFCK->Fill(convertGeVToKeV(esum.eTime[0]));             //in keV
        hELCSMaxCK[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));  //in keV
        if (convertGeVToKeV(esum.eTime[0]) < 10.)
          hXYhitsLELCK[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        else if (convertGeVToKeV(esum.eTime[0]) > 50.)
          hXYhitsHELCK[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        if (isPWafer) {
          hNHxELossCSMaxCK->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxELCSMaxCK[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
          hNHxXYhitsCK[hinfo.layer - 1]->Fill(hinfo.x, hinfo.y);
        }
        if (isFWafer) {
          hHxELossCSMaxCK->Fill(convertGeVToKeV(esum.eTime[0]));
          hHxELCSMaxCK[hinfo.layer - 1]->Fill(convertGeVToKeV(esum.eTime[0]));
        }
      }
    }
  }

  map_hits.clear();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalCellHitSum::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simtrack", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("simhits", edm::InputTag("g4SimHits", "HGCHitsEE"));
  desc.add<std::string>("detector", "HGCalEESensitive");
  desc.add<edm::FileInPath>("geometryFileName", edm::FileInPath("Validation/HGCalValidation/data/wafer_v17.csv"));
  descriptions.add("hgcalCellHitSumEE", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalCellHitSum);
