// -*- C++ -*-
//
// Package:    HGCalValidation/HGCalMTRecoStudy
// Class:      HGCalMTRecoStudy
//
/**\class HGCalMTRecoStudy HGCalMTRecoStudy.cc Validation/HGCalValidation/test/HGCalMTRecoStudy.cc
// Derived from : HGCalRecHitStudy.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Indranil Das
//         Created:  Thu, 06 Oct 2022 06:18:11 GMT
//
//

// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"

//
// class declaration
//

class HGCalMTRecoStudy : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  //Implemented following Validation/HGCalValidation/test/HGCalRecHitStudy.cc
  explicit HGCalMTRecoStudy(const edm::ParameterSet &);
  ~HGCalMTRecoStudy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  // ----------member data ---------------------------
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  hgcal::RecHitTools rhtools_;
  const std::string nameDetector_;
  const edm::EDGetTokenT<HGCRecHitCollection> recHitSource_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> tok_hgcaldd_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcGeom_;

  const std::string layers_;
  std::vector<Int_t> layerList;

  // For rechittool z positions. The 0 and 1 are for -ve and +ve, respectively.
  TH1D *hEF;
  TH1D *hECN;
  TH1D *hECK;
  TH1D *hESc;

  std::vector<TH1D *> hELossLayer0;
  std::vector<TH1D *> hELossLayer1;
  std::vector<TH1D *> hNCellsLayer0;
  std::vector<TH1D *> hNCellsLayer1;

  std::vector<TH2D *> hXYhitsF0;
  std::vector<TH2D *> hXYhitsCN0;
  std::vector<TH2D *> hXYhitsCK0;
  std::vector<TH2D *> hXYhitsB0;

  std::vector<TH2D *> hXYhitsF1;
  std::vector<TH2D *> hXYhitsCN1;
  std::vector<TH2D *> hXYhitsCK1;
  std::vector<TH2D *> hXYhitsB1;

  std::vector<TH2D *> hEPhitsF0;
  std::vector<TH2D *> hEPhitsCN0;
  std::vector<TH2D *> hEPhitsCK0;
  std::vector<TH2D *> hEPhitsB0;

  std::vector<TH2D *> hEPhitsF1;
  std::vector<TH2D *> hEPhitsCN1;
  std::vector<TH2D *> hEPhitsCK1;
  std::vector<TH2D *> hEPhitsB1;

  std::vector<TH2D *> hXYFailhitsF0;
  std::vector<TH2D *> hXYFailhitsCN0;
  std::vector<TH2D *> hXYFailhitsCK0;
  std::vector<TH2D *> hXYFailhitsB0;

  std::vector<TH2D *> hXYFailhitsF1;
  std::vector<TH2D *> hXYFailhitsCN1;
  std::vector<TH2D *> hXYFailhitsCK1;
  std::vector<TH2D *> hXYFailhitsB1;

  std::vector<TH2D *> hEPFailhitsF0;
  std::vector<TH2D *> hEPFailhitsCN0;
  std::vector<TH2D *> hEPFailhitsCK0;
  std::vector<TH2D *> hEPFailhitsB0;

  std::vector<TH2D *> hEPFailhitsF1;
  std::vector<TH2D *> hEPFailhitsCN1;
  std::vector<TH2D *> hEPFailhitsCK1;
  std::vector<TH2D *> hEPFailhitsB1;

  std::vector<TH1D *> hELossLayerF0;
  std::vector<TH1D *> hELossLayerCN0;
  std::vector<TH1D *> hELossLayerCK0;
  std::vector<TH1D *> hELossLayerB0;

  std::vector<TH1D *> hELossLayerF1;
  std::vector<TH1D *> hELossLayerCN1;
  std::vector<TH1D *> hELossLayerCK1;
  std::vector<TH1D *> hELossLayerB1;

  // For rechittool z positions. The 0 and 1 are for -ve and +ve, respectively.
  std::vector<TGraph *> grXYhitsF0;
  std::vector<TGraph *> grXYhitsCN0;
  std::vector<TGraph *> grXYhitsCK0;
  std::vector<TGraph *> grXYhitsAR0;
  std::vector<TGraph *> grXYhitsB0;
  int ixyF0[50], ixyCN0[50], ixyCK0[50], ixyAR0[50], ixyB0[50];

  std::vector<TGraph *> grXYhitsF1;
  std::vector<TGraph *> grXYhitsCN1;
  std::vector<TGraph *> grXYhitsCK1;
  std::vector<TGraph *> grXYhitsAR1;
  std::vector<TGraph *> grXYhitsB1;
  int ixyF1[50], ixyCN1[50], ixyCK1[50], ixyAR1[50], ixyB1[50];
  /////////////////////////////////

  // For rechittool z positions. The 0 and 1 are for -ve and +ve, respectively.
  std::vector<TGraph *> grEtaPhihitsF0;
  std::vector<TGraph *> grEtaPhihitsCN0;
  std::vector<TGraph *> grEtaPhihitsCK0;
  std::vector<TGraph *> grEtaPhihitsB0;
  int iepF0[50], iepCN0[50], iepCK0[50], iepB0[50];

  std::vector<TGraph *> grEtaPhihitsF1;
  std::vector<TGraph *> grEtaPhihitsCN1;
  std::vector<TGraph *> grEtaPhihitsCK1;
  std::vector<TGraph *> grEtaPhihitsB1;
  int iepF1[50], iepCN1[50], iepCK1[50], iepB1[50];
};

//
// constructors and destructor
//
HGCalMTRecoStudy::HGCalMTRecoStudy(const edm::ParameterSet &iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("detectorName")),
      recHitSource_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("source"))),
      tok_hgcaldd_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      tok_hgcGeom_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameDetector_})),
      layers_(iConfig.getParameter<std::string>("layerList")) {
  layerList.clear();

  if (layers_.find("-") != std::string::npos) {
    std::vector<std::string> tokens;
    std::stringstream check1(layers_);
    std::string intermediate;
    while (getline(check1, intermediate, '-'))
      tokens.push_back(intermediate);
    int minLayer = (stoi(tokens[0]) < 1) ? 1 : stoi(tokens[0]);
    int maxLayer = (stoi(tokens[1]) > 47) ? 47 : stoi(tokens[1]);
    for (int i = minLayer; i <= maxLayer; i++) {
      layerList.push_back(i);
      //std::cout << tokens[i] << '\n';
    }
    tokens.clear();

  } else if (layers_.find(",") != std::string::npos) {
    std::vector<std::string> tokens;
    std::stringstream check1(layers_);
    std::string intermediate;
    while (getline(check1, intermediate, ','))
      tokens.push_back(intermediate);
    for (unsigned int i = 0; i < tokens.size(); i++) {
      if (stoi(tokens[i]) >= 1 and stoi(tokens[i]) <= 47)
        layerList.push_back(stoi(tokens[i]));
      //std::cout << tokens[i] << '\n';
    }
    tokens.clear();

  } else {
    if (stoi(layers_) >= 1 and stoi(layers_) <= 47)
      layerList.push_back(stoi(layers_));
  }

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  hEF = fs->make<TH1D>("hEF", "hEF", 1000, 0., 50.);
  hECN = fs->make<TH1D>("hECN", "hECN", 1000, 0., 50.);
  hECK = fs->make<TH1D>("hECK", "hECK", 1000, 0., 50.);
  hESc = fs->make<TH1D>("hESc", "hESc", 1000, 0., 50.);

  for (unsigned int i = 0; i < 47; i++) {
    hELossLayer0.emplace_back(
        fs->make<TH1D>(Form("hELossLayer0_%02d", i + 1), Form("Rec:ELoss0 for layer %d", i + 1), 500, 0., 5000.));
    hELossLayer1.emplace_back(
        fs->make<TH1D>(Form("hELossLayer1_%02d", i + 1), Form("Rec:ELoss1 for layer %d", i + 1), 500, 0., 5000.));
    hNCellsLayer0.emplace_back(
        fs->make<TH1D>(Form("hNCellsLayer0_%02d", i + 1), Form("Rec:NCells0 for layer %d", i + 1), 200, -0.5, 199.5));
    hNCellsLayer1.emplace_back(
        fs->make<TH1D>(Form("hNCellsLayer1_%02d", i + 1), Form("Rec:NCells1 for layer %d", i + 1), 200, -0.5, 199.5));
  }

  for (unsigned int i = 0; i < layerList.size(); i++) {
    hXYhitsF0.emplace_back(fs->make<TH2D>(Form("hXYhitsF0_layer_%02d", layerList[i]),
                                          Form("Rec:HitsF0 in XY for layer %d", layerList[i]),
                                          600,
                                          -300.,
                                          300.,
                                          600,
                                          -300.,
                                          300.));
    hXYhitsCN0.emplace_back(fs->make<TH2D>(Form("hXYhitsCN0_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCN0 in XY for layer %d", layerList[i]),
                                           600,
                                           -300.,
                                           300.,
                                           600,
                                           -300.,
                                           300.));
    hXYhitsCK0.emplace_back(fs->make<TH2D>(Form("hXYhitsCK0_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCK0 in XY for layer %d", layerList[i]),
                                           600,
                                           -300.,
                                           300.,
                                           600,
                                           -300.,
                                           300.));
    hXYhitsB0.emplace_back(fs->make<TH2D>(Form("hXYhitsB0_layer_%02d", layerList[i]),
                                          Form("Rec:HitsB0 in XY for layer %d", layerList[i]),
                                          600,
                                          -300.,
                                          300.,
                                          600,
                                          -300.,
                                          300.));
    hXYhitsF1.emplace_back(fs->make<TH2D>(Form("hXYhitsF1_layer_%02d", layerList[i]),
                                          Form("Rec:HitsF1 in XY for layer %d", layerList[i]),
                                          600,
                                          -300.,
                                          300.,
                                          600,
                                          -300.,
                                          300.));
    hXYhitsCN1.emplace_back(fs->make<TH2D>(Form("hXYhitsCN1_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCN1 in XY for layer %d", layerList[i]),
                                           600,
                                           -300.,
                                           300.,
                                           600,
                                           -300.,
                                           300.));
    hXYhitsCK1.emplace_back(fs->make<TH2D>(Form("hXYhitsCK1_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCK1 in XY for layer %d", layerList[i]),
                                           600,
                                           -300.,
                                           300.,
                                           600,
                                           -300.,
                                           300.));
    hXYhitsB1.emplace_back(fs->make<TH2D>(Form("hXYhitsB1_layer_%02d", layerList[i]),
                                          Form("Rec:HitsB1 in XY for layer %d", layerList[i]),
                                          600,
                                          -300.,
                                          300.,
                                          600,
                                          -300.,
                                          300.));

    hEPhitsF0.emplace_back(fs->make<TH2D>(Form("hEPhitsF0_layer_%02d", layerList[i]),
                                          Form("Rec:HitsF0 in EP for layer %d", layerList[i]),
                                          640,
                                          -3.2,
                                          3.2,
                                          640,
                                          -3.2,
                                          3.2));
    hEPhitsCN0.emplace_back(fs->make<TH2D>(Form("hEPhitsCN0_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCN0 in EP for layer %d", layerList[i]),
                                           640,
                                           -3.2,
                                           3.2,
                                           640,
                                           -3.2,
                                           3.2));
    hEPhitsCK0.emplace_back(fs->make<TH2D>(Form("hEPhitsCK0_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCK0 in EP for layer %d", layerList[i]),
                                           640,
                                           -3.2,
                                           3.2,
                                           640,
                                           -3.2,
                                           3.2));
    hEPhitsB0.emplace_back(fs->make<TH2D>(Form("hEPhitsB0_layer_%02d", layerList[i]),
                                          Form("Rec:HitsB0 in EP for layer %d", layerList[i]),
                                          640,
                                          -3.2,
                                          3.2,
                                          640,
                                          -3.2,
                                          3.2));
    hEPhitsF1.emplace_back(fs->make<TH2D>(Form("hEPhitsF1_layer_%02d", layerList[i]),
                                          Form("Rec:HitsF1 in EP for layer %d", layerList[i]),
                                          640,
                                          -3.2,
                                          3.2,
                                          640,
                                          -3.2,
                                          3.2));
    hEPhitsCN1.emplace_back(fs->make<TH2D>(Form("hEPhitsCN1_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCN1 in EP for layer %d", layerList[i]),
                                           640,
                                           -3.2,
                                           3.2,
                                           640,
                                           -3.2,
                                           3.2));
    hEPhitsCK1.emplace_back(fs->make<TH2D>(Form("hEPhitsCK1_layer_%02d", layerList[i]),
                                           Form("Rec:HitsCK1 in EP for layer %d", layerList[i]),
                                           640,
                                           -3.2,
                                           3.2,
                                           640,
                                           -3.2,
                                           3.2));
    hEPhitsB1.emplace_back(fs->make<TH2D>(Form("hEPhitsB1_layer_%02d", layerList[i]),
                                          Form("Rec:HitsB1 in EP for layer %d", layerList[i]),
                                          640,
                                          -3.2,
                                          3.2,
                                          640,
                                          -3.2,
                                          3.2));

    hXYFailhitsF0.emplace_back(fs->make<TH2D>(Form("hXYFailhitsF0_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsF0 in XY for layer %d", layerList[i]),
                                              600,
                                              -300.,
                                              300.,
                                              600,
                                              -300.,
                                              300.));
    hXYFailhitsCN0.emplace_back(fs->make<TH2D>(Form("hXYFailhitsCN0_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCN0 in XY for layer %d", layerList[i]),
                                               600,
                                               -300.,
                                               300.,
                                               600,
                                               -300.,
                                               300.));
    hXYFailhitsCK0.emplace_back(fs->make<TH2D>(Form("hXYFailhitsCK0_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCK0 in XY for layer %d", layerList[i]),
                                               600,
                                               -300.,
                                               300.,
                                               600,
                                               -300.,
                                               300.));
    hXYFailhitsB0.emplace_back(fs->make<TH2D>(Form("hXYFailhitsB0_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsB0 in XY for layer %d", layerList[i]),
                                              600,
                                              -300.,
                                              300.,
                                              600,
                                              -300.,
                                              300.));
    hXYFailhitsF1.emplace_back(fs->make<TH2D>(Form("hXYFailhitsF1_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsF1 in XY for layer %d", layerList[i]),
                                              600,
                                              -300.,
                                              300.,
                                              600,
                                              -300.,
                                              300.));
    hXYFailhitsCN1.emplace_back(fs->make<TH2D>(Form("hXYFailhitsCN1_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCN1 in XY for layer %d", layerList[i]),
                                               600,
                                               -300.,
                                               300.,
                                               600,
                                               -300.,
                                               300.));
    hXYFailhitsCK1.emplace_back(fs->make<TH2D>(Form("hXYFailhitsCK1_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCK1 in XY for layer %d", layerList[i]),
                                               600,
                                               -300.,
                                               300.,
                                               600,
                                               -300.,
                                               300.));
    hXYFailhitsB1.emplace_back(fs->make<TH2D>(Form("hXYFailhitsB1_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsB1 in XY for layer %d", layerList[i]),
                                              600,
                                              -300.,
                                              300.,
                                              600,
                                              -300.,
                                              300.));

    hEPFailhitsF0.emplace_back(fs->make<TH2D>(Form("hEPFailhitsF0_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsF0 in EP for layer %d", layerList[i]),
                                              640,
                                              -3.2,
                                              3.2,
                                              640,
                                              -3.2,
                                              3.2));
    hEPFailhitsCN0.emplace_back(fs->make<TH2D>(Form("hEPFailhitsCN0_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCN0 in EP for layer %d", layerList[i]),
                                               640,
                                               -3.2,
                                               3.2,
                                               640,
                                               -3.2,
                                               3.2));
    hEPFailhitsCK0.emplace_back(fs->make<TH2D>(Form("hEPFailhitsCK0_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCK0 in EP for layer %d", layerList[i]),
                                               640,
                                               -3.2,
                                               3.2,
                                               640,
                                               -3.2,
                                               3.2));
    hEPFailhitsB0.emplace_back(fs->make<TH2D>(Form("hEPFailhitsB0_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsB0 in EP for layer %d", layerList[i]),
                                              640,
                                              -3.2,
                                              3.2,
                                              640,
                                              -3.2,
                                              3.2));
    hEPFailhitsF1.emplace_back(fs->make<TH2D>(Form("hEPFailhitsF1_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsF1 in EP for layer %d", layerList[i]),
                                              640,
                                              -3.2,
                                              3.2,
                                              640,
                                              -3.2,
                                              3.2));
    hEPFailhitsCN1.emplace_back(fs->make<TH2D>(Form("hEPFailhitsCN1_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCN1 in EP for layer %d", layerList[i]),
                                               640,
                                               -3.2,
                                               3.2,
                                               640,
                                               -3.2,
                                               3.2));
    hEPFailhitsCK1.emplace_back(fs->make<TH2D>(Form("hEPFailhitsCK1_layer_%02d", layerList[i]),
                                               Form("Rec:FailhitsCK1 in EP for layer %d", layerList[i]),
                                               640,
                                               -3.2,
                                               3.2,
                                               640,
                                               -3.2,
                                               3.2));
    hEPFailhitsB1.emplace_back(fs->make<TH2D>(Form("hEPFailhitsB1_layer_%02d", layerList[i]),
                                              Form("Rec:FailhitsB1 in EP for layer %d", layerList[i]),
                                              640,
                                              -3.2,
                                              3.2,
                                              640,
                                              -3.2,
                                              3.2));

    hELossLayerF0.emplace_back(fs->make<TH1D>(Form("hELossLayerF0_layer_%02d", layerList[i]),
                                              Form("Rec:ELossF0 in XY for layer %d", layerList[i]),
                                              1000,
                                              0.,
                                              1000.));
    hELossLayerCN0.emplace_back(fs->make<TH1D>(Form("hELossLayerCN0_layer_%02d", layerList[i]),
                                               Form("Rec:ELossCN0 in XY for layer %d", layerList[i]),
                                               1000,
                                               0.,
                                               1000.));
    hELossLayerCK0.emplace_back(fs->make<TH1D>(Form("hELossLayerCK0_layer_%02d", layerList[i]),
                                               Form("Rec:ELossCK0 in XY for layer %d", layerList[i]),
                                               1000,
                                               0.,
                                               1000.));
    hELossLayerB0.emplace_back(fs->make<TH1D>(Form("hELossLayerB0_layer_%02d", layerList[i]),
                                              Form("Rec:ELossB0 in XY for layer %d", layerList[i]),
                                              1000,
                                              0.,
                                              1000.));
    hELossLayerF1.emplace_back(fs->make<TH1D>(Form("hELossLayerF1_layer_%02d", layerList[i]),
                                              Form("Rec:ELossF1 in XY for layer %d", layerList[i]),
                                              1000,
                                              0.,
                                              1000.));
    hELossLayerCN1.emplace_back(fs->make<TH1D>(Form("hELossLayerCN1_layer_%02d", layerList[i]),
                                               Form("Rec:ELossCN1 in XY for layer %d", layerList[i]),
                                               1000,
                                               0.,
                                               1000.));
    hELossLayerCK1.emplace_back(fs->make<TH1D>(Form("hELossLayerCK1_layer_%02d", layerList[i]),
                                               Form("ELossCK1 in XY for layer %d", layerList[i]),
                                               1000,
                                               0.,
                                               1000.));
    hELossLayerB1.emplace_back(fs->make<TH1D>(Form("hELossLayerB1_layer_%02d", layerList[i]),
                                              Form("Rec:ELossB1 in XY for layer %d", layerList[i]),
                                              1000,
                                              0.,
                                              1000.));
  }

  for (unsigned int i = 0; i < layerList.size(); i++) {
    grXYhitsF0.emplace_back(fs->make<TGraph>(0));
    grXYhitsF0[i]->SetNameTitle(Form("grXYhitsF0_layer_%02d", layerList[i]),
                                Form("Rec:HitsF0 in XY for layer %d", layerList[i]));
    grXYhitsCN0.emplace_back(fs->make<TGraph>(0));
    grXYhitsCN0[i]->SetNameTitle(Form("grXYhitsCN0_layer_%02d", layerList[i]),
                                 Form("Rec:HitsCN0 in XY for layer %d", layerList[i]));
    grXYhitsCK0.emplace_back(fs->make<TGraph>(0));
    grXYhitsCK0[i]->SetNameTitle(Form("grXYhitsCK0_layer_%02d", layerList[i]),
                                 Form("Rec:HitsCK0 in XY for layer %d", layerList[i]));
    grXYhitsB0.emplace_back(fs->make<TGraph>(0));
    grXYhitsB0[i]->SetNameTitle(Form("grXYhitsB0_layer_%02d", layerList[i]),
                                Form("Rec:HitsB0 in XY for layer %d", layerList[i]));
    grXYhitsAR0.emplace_back(fs->make<TGraph>(0));
    grXYhitsAR0[i]->SetNameTitle(Form("grXYhitsAR0_layer_%02d", layerList[i]),
                                 Form("Rec:HitsAR0 in XY for layer %d", layerList[i]));
    ixyF0[i] = 0;
    ixyCN0[i] = 0;
    ixyCK0[i] = 0;
    ixyB0[i] = 0;
    ixyAR0[i] = 0;

    grXYhitsF1.emplace_back(fs->make<TGraph>(0));
    grXYhitsF1[i]->SetNameTitle(Form("grXYhitsF1_layer_%02d", layerList[i]),
                                Form("Rec:HitsF1 in XY for layer %d", layerList[i]));
    grXYhitsCN1.emplace_back(fs->make<TGraph>(0));
    grXYhitsCN1[i]->SetNameTitle(Form("grXYhitsCN1_layer_%02d", layerList[i]),
                                 Form("Rec:HitsCN1 in XY for layer %d", layerList[i]));
    grXYhitsCK1.emplace_back(fs->make<TGraph>(0));
    grXYhitsCK1[i]->SetNameTitle(Form("grXYhitsCK1_layer_%02d", layerList[i]),
                                 Form("Rec:HitsCK1 in XY for layer %d", layerList[i]));
    grXYhitsB1.emplace_back(fs->make<TGraph>(0));
    grXYhitsB1[i]->SetNameTitle(Form("grXYhitsB1_layer_%02d", layerList[i]),
                                Form("Rec:HitsB1 in XY for layer %d", layerList[i]));
    grXYhitsAR1.emplace_back(fs->make<TGraph>(0));
    grXYhitsAR1[i]->SetNameTitle(Form("grXYhitsAR1_layer_%02d", layerList[i]),
                                 Form("Rec:HitsAR1 in XY for layer %d", layerList[i]));
    ixyF1[i] = 0;
    ixyCN1[i] = 0;
    ixyCK1[i] = 0;
    ixyB1[i] = 0;
    ixyAR1[i] = 0;

    grEtaPhihitsF0.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsF0[i]->SetNameTitle(Form("grEtaPhihitsF0_layer_%02d", layerList[i]),
                                    Form("Rec:HitsF0 in XY for layer %d", layerList[i]));
    grEtaPhihitsCN0.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsCN0[i]->SetNameTitle(Form("grEtaPhihitsCN0_layer_%02d", layerList[i]),
                                     Form("Rec:HitsCN0 in XY for layer %d", layerList[i]));
    grEtaPhihitsCK0.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsCK0[i]->SetNameTitle(Form("grEtaPhihitsCK0_layer_%02d", layerList[i]),
                                     Form("Rec:HitsCK0 in XY for layer %d", layerList[i]));
    grEtaPhihitsB0.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsB0[i]->SetNameTitle(Form("grEtaPhihitsB0_layer_%02d", layerList[i]),
                                    Form("Rec:HitsB0 in XY for layer %d", layerList[i]));
    iepF0[i] = 0;
    iepCN0[i] = 0;
    iepCK0[i] = 0;
    iepB0[i] = 0;

    grEtaPhihitsF1.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsF1[i]->SetNameTitle(Form("grEtaPhihitsF1_layer_%02d", layerList[i]),
                                    Form("Rec:HitsF1 in XY for layer %d", layerList[i]));
    grEtaPhihitsCN1.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsCN1[i]->SetNameTitle(Form("grEtaPhihitsCN1_layer_%02d", layerList[i]),
                                     Form("Rec:HitsCN1 in XY for layer %d", layerList[i]));
    grEtaPhihitsCK1.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsCK1[i]->SetNameTitle(Form("grEtaPhihitsCK1_layer_%02d", layerList[i]),
                                     Form("Rec:HitsCK1 in XY for layer %d", layerList[i]));
    grEtaPhihitsB1.emplace_back(fs->make<TGraph>(0));
    grEtaPhihitsB1[i]->SetNameTitle(Form("grEtaPhihitsB1_layer_%02d", layerList[i]),
                                    Form("Rec:HitsB1 in XY for layer %d", layerList[i]));
    iepF1[i] = 0;
    iepCN1[i] = 0;
    iepCK1[i] = 0;
    iepB1[i] = 0;
  }

  caloGeomToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
}

//
// member functions
//

// ------------ method called for each event  ------------
void HGCalMTRecoStudy::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  const CaloGeometry &geomCalo = iSetup.getData(caloGeomToken_);
  rhtools_.setGeometry(geomCalo);

  int verbosity_ = 0;
  double ElossLayer0[47], ElossLayer1[47];
  int nCellsLayer0[47], nCellsLayer1[47];
  for (int ilayer = 0; ilayer < 47; ilayer++) {
    ElossLayer0[ilayer] = ElossLayer1[ilayer] = 0.0;
    nCellsLayer0[ilayer] = nCellsLayer1[ilayer] = 0;
  }

  const edm::ESHandle<HGCalGeometry> &geom = iSetup.getHandle(tok_hgcGeom_);
  if (!geom.isValid())
    edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
  const HGCalGeometry *geom0 = geom.product();

  const edm::Handle<HGCRecHitCollection> &theRecHitContainers = iEvent.getHandle(recHitSource_);
  if (theRecHitContainers.isValid()) {
    if (verbosity_ > 0)
      edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << theRecHitContainers->size() << " element(s)";

    for (const auto &it : *(theRecHitContainers.product())) {
      DetId detId = it.id();
      GlobalPoint global1 = rhtools_.getPosition(detId);
      double energy = it.energy() * 1.e3;

      std::vector<int>::iterator ilyr =
          std::find(layerList.begin(), layerList.end(), rhtools_.getLayerWithOffset(detId));

      if (geom0->topology().valid(detId)) {
        if (rhtools_.isSilicon(detId)) {
          HGCSiliconDetId id(it.id());

          if (id.type() == HGCSiliconDetId::HGCalFine) {
            hEF->Fill(energy);

            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0) {
                grXYhitsF0[il]->SetPoint(ixyF0[il]++, global1.x(), global1.y());
                grEtaPhihitsF0[il]->SetPoint(iepF0[il]++, global1.eta(), global1.phi());
                hXYhitsF0[il]->Fill(global1.x(), global1.y());
                hEPhitsF0[il]->Fill(global1.eta(), global1.phi());
                hELossLayerF0[il]->Fill(energy);
              } else {
                grXYhitsF1[il]->SetPoint(ixyF1[il]++, global1.x(), global1.y());
                grEtaPhihitsF1[il]->SetPoint(iepF1[il]++, global1.eta(), global1.phi());
                hXYhitsF1[il]->Fill(global1.x(), global1.y());
                hEPhitsF1[il]->Fill(global1.eta(), global1.phi());
                hELossLayerF1[il]->Fill(energy);
              }
            }  //isvalid layer
          }
          if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
            hECN->Fill(energy);

            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0) {
                grXYhitsCN0[il]->SetPoint(ixyCN0[il]++, global1.x(), global1.y());
                grEtaPhihitsCN0[il]->SetPoint(iepCN0[il]++, global1.eta(), global1.phi());
                hXYhitsCN0[il]->Fill(global1.x(), global1.y());
                hEPhitsCN0[il]->Fill(global1.eta(), global1.phi());
                hELossLayerCN0[il]->Fill(energy);
              } else {
                grXYhitsCN1[il]->SetPoint(ixyCN1[il]++, global1.x(), global1.y());
                grEtaPhihitsCN1[il]->SetPoint(iepCN1[il]++, global1.eta(), global1.phi());
                hXYhitsCN1[il]->Fill(global1.x(), global1.y());
                hEPhitsCN1[il]->Fill(global1.eta(), global1.phi());
                hELossLayerCN1[il]->Fill(energy);
              }
            }  //isvalid layer
          }
          if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {  //case 2 :
            hECK->Fill(energy);

            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0) {
                grXYhitsCK0[il]->SetPoint(ixyCK0[il]++, global1.x(), global1.y());
                grEtaPhihitsCK0[il]->SetPoint(iepCK0[il]++, global1.eta(), global1.phi());
                hXYhitsCK0[il]->Fill(global1.x(), global1.y());
                hEPhitsCK0[il]->Fill(global1.eta(), global1.phi());
                hELossLayerCK0[il]->Fill(energy);
              } else {
                grXYhitsCK1[il]->SetPoint(ixyCK1[il]++, global1.x(), global1.y());
                grEtaPhihitsCK1[il]->SetPoint(iepCK1[il]++, global1.eta(), global1.phi());
                hXYhitsCK1[il]->Fill(global1.x(), global1.y());
                hEPhitsCK1[il]->Fill(global1.eta(), global1.phi());
                hELossLayerCK1[il]->Fill(energy);
              }
            }  //isvalid layer
          }
          //The following line by Pruthvi to number the cells with id0 and detId
          if (rhtools_.getCell(detId).first + rhtools_.getCell(detId).second <= 2) {
            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0)
                grXYhitsAR0[il]->SetPoint(ixyAR0[il]++, global1.x(), global1.y());
              else
                grXYhitsAR1[il]->SetPoint(ixyAR1[il]++, global1.x(), global1.y());
            }  //isvalid layer
          }
        } else if (rhtools_.isScintillator(detId)) {
          //HGCScintillatorDetId id(itHit->id());
          //int il = rhtools_.getLayerWithOffset(detId);
          hESc->Fill(energy);

          if (ilyr != layerList.cend()) {
            int il = std::distance(layerList.begin(), ilyr);
            if (global1.z() < 0.0) {
              grXYhitsB0[il]->SetPoint(ixyB0[il]++, global1.x(), global1.y());
              grEtaPhihitsB0[il]->SetPoint(iepB0[il]++, global1.eta(), global1.phi());
              hXYhitsB0[il]->Fill(global1.x(), global1.y());
              hEPhitsB0[il]->Fill(global1.eta(), global1.phi());
              hELossLayerB0[il]->Fill(energy);
            } else {
              grXYhitsB1[il]->SetPoint(ixyB1[il]++, global1.x(), global1.y());
              grEtaPhihitsB1[il]->SetPoint(iepB1[il]++, global1.eta(), global1.phi());
              hXYhitsB1[il]->Fill(global1.x(), global1.y());
              hEPhitsB1[il]->Fill(global1.eta(), global1.phi());
              hELossLayerB1[il]->Fill(energy);
            }
          }  //isvalid layer

        }  //Silicon or scintillator

        int ilayer = rhtools_.getLayerWithOffset(detId) - 1;
        if (global1.z() < 0.0) {
          ElossLayer0[ilayer] += energy;
          nCellsLayer0[ilayer]++;
        } else {
          ElossLayer1[ilayer] += energy;
          nCellsLayer1[ilayer]++;
        }

        ///.................
      } else {  //valid topology else invalid

        if (rhtools_.isSilicon(detId)) {
          HGCSiliconDetId id(it.id());
          //int il = rhtools_.getLayerWithOffset(detId);
          if (id.type() == HGCSiliconDetId::HGCalFine) {
            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0) {
                hXYFailhitsF0[il]->Fill(global1.x(), global1.y());
                hEPFailhitsF0[il]->Fill(global1.eta(), global1.phi());
              } else {
                hXYFailhitsF1[il]->Fill(global1.x(), global1.y());
                hEPFailhitsF1[il]->Fill(global1.eta(), global1.phi());
              }
            }  //isvalid layer
          }
          if (id.type() == HGCSiliconDetId::HGCalCoarseThin) {
            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0) {
                hXYFailhitsCN0[il]->Fill(global1.x(), global1.y());
                hEPFailhitsCN0[il]->Fill(global1.eta(), global1.phi());
              } else {
                hXYFailhitsCN1[il]->Fill(global1.x(), global1.y());
                hEPFailhitsCN1[il]->Fill(global1.eta(), global1.phi());
              }
            }  //isvalid layer
          }
          if (id.type() == HGCSiliconDetId::HGCalCoarseThick) {  //case 2 :

            if (ilyr != layerList.cend()) {
              int il = std::distance(layerList.begin(), ilyr);
              if (global1.z() < 0.0) {
                hXYFailhitsCK0[il]->Fill(global1.x(), global1.y());
                hEPFailhitsCK0[il]->Fill(global1.eta(), global1.phi());
              } else {
                hXYFailhitsCK1[il]->Fill(global1.x(), global1.y());
                hEPFailhitsCK1[il]->Fill(global1.eta(), global1.phi());
              }
            }  //isvalid layer
          }
        } else if (rhtools_.isScintillator(detId)) {
          //int il = rhtools_.getLayerWithOffset(detId);
          if (ilyr != layerList.cend()) {
            int il = std::distance(layerList.begin(), ilyr);
            if (global1.z() < 0.0) {
              hXYFailhitsB0[il]->Fill(global1.x(), global1.y());
              hEPFailhitsB0[il]->Fill(global1.eta(), global1.phi());
            } else {
              hXYFailhitsB1[il]->Fill(global1.x(), global1.y());
              hEPFailhitsB1[il]->Fill(global1.eta(), global1.phi());
            }
          }  //isvalid layer

        }  //Silicon or scintillator

      }  //invalid topology
    }    //loop over iterator
  }      //is Valid container

  for (int i = 0; i < 47; i++) {
    if (ElossLayer0[i] > 0.0)
      hELossLayer0[i]->Fill(ElossLayer0[i]);
    if (nCellsLayer0[i] > 0)
      hNCellsLayer0[i]->Fill(nCellsLayer0[i]);

    if (ElossLayer1[i] > 0.0)
      hELossLayer1[i]->Fill(ElossLayer1[i]);
    if (nCellsLayer1[i] > 0)
      hNCellsLayer1[i]->Fill(nCellsLayer1[i]);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalMTRecoStudy::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detectorName", "HGCalEESensitive");
  desc.add<edm::InputTag>("source", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<std::string>("layerList", "1");
  descriptions.add("hgcalMTRecoStudy", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalMTRecoStudy);
