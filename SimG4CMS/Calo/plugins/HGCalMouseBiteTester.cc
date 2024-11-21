// -*- C++ -*-
//
// Package:    HGCalMouseBiteTester
// Class:      HGCalMouseBiteTester
//
/**\class HGCalMouseBiteTester HGCalMouseBiteTester.cc
 plugins/HGCalMouseBiteTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee, Pruthvi Suryadevara
//         Created:  Mon 2023/11/30
//
//

// system include files
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
//#include <chrono>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include <CLHEP/Vector/ThreeVector.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellUV.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "SimG4CMS/Calo/interface/HGCMouseBite.h"
#include "SimG4CMS/Calo/interface/HGCGuardRing.h"
#include "SimG4CMS/Calo/interface/HGCGuardRingPartial.h"
#include "G4ThreeVector.hh"

class HGCalMouseBiteTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalMouseBiteTester(const edm::ParameterSet&);
  ~HGCalMouseBiteTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_;
  const int waferU_;
  const int waferV_;
  const int nTrials_;
  const int layer_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  std::ofstream outputFile;
};

HGCalMouseBiteTester::HGCalMouseBiteTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("nameSense")),
      waferU_(iC.getParameter<int>("waferU")),
      waferV_(iC.getParameter<int>("waferV")),
      nTrials_(iC.getParameter<int>("numbberOfTrials")),
      layer_(iC.getParameter<int>("layer")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeom") << "Test Guard_Ring for wafer in layer" << layer_ << " U " << waferU_ << " V "
                                << waferV_ << " with " << nTrials_ << " trials";

  outputFile.open("full1.csv");
  if (!outputFile.is_open()) {
    edm::LogError("HGCalGeom") << "Could not open output file.";
  } else {
    outputFile << "x,y,u,v,\n";
  }
}

void HGCalMouseBiteTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<int>("waferU", 1);
  desc.add<int>("waferV", 9);
  desc.add<int>("numbberOfTrials", 1000000);
  desc.add<int>("layer", 1);
  descriptions.add("hgcalMouseBiteTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalMouseBiteTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgcons_ = iSetup.getData(dddToken_);
  double waferSize_(hgcons_.waferSize(false));
  int zside(1);
  int layertype = hgcons_.layerType(layer_);
  int frontBack = HGCalTypes::layerFrontBack(layertype);
  const std::vector<double> angle_{90.0, 30.0};
  int index = HGCalWaferIndex::waferIndex(layer_, waferU_, waferV_);
  int partialType_ = HGCalWaferType::getPartial(index, hgcons_.getParameter()->waferInfoMap_);
  int orient = HGCalWaferType::getOrient(index, hgcons_.getParameter()->waferInfoMap_);
  int placeIndex_ = HGCalCell::cellPlacementIndex(zside, frontBack, orient);
  int waferType_ = HGCalWaferType::getType(index, hgcons_.getParameter()->waferInfoMap_);
  double mouseBiteCut_ = hgcons_.mouseBite(false);
  bool v17OrLess = hgcons_.v17OrLess();
  HGCGuardRing guardRing_(hgcons_);
  HGCGuardRingPartial guardRingPartial_(hgcons_);
  HGCMouseBite mouseBite_(hgcons_, angle_, (waferSize_ * tan(30.0 * CLHEP::deg) - mouseBiteCut_), true);
  const int nFine(12), nCoarse(8);
  double r2 = 0.5 * waferSize_;
  double R2 = 2 * r2 / sqrt(3);
  int nCells = (waferType_ == 0) ? nFine : nCoarse;
  std::cout << "start" << std::endl;
  HGCalCellUV wafer(waferSize_, 0.0, nFine, nCoarse);
  HGCalCell wafer2(waferSize_, nFine, nCoarse);
  std::pair<double, double> xy = hgcons_.waferPosition(layer_, waferU_, waferV_, false, false);
  double x0 = (zside > 0) ? xy.first : -xy.first;
  double y0 = xy.second;
  std::ofstream guard_ring("Guard_ring.csv");
  std::ofstream guard_ring_partial("Guard_ring_partial.csv");
  std::ofstream mouse_bite("Mouse_bite.csv");
  std::ofstream selected("Selected.csv");
  edm::LogVerbatim("HGCalGeom") << "\nHGCalMouseBiteTester:: nCells " << nCells << " FrontBack " << frontBack
                                << " Wafer Size " << waferSize_ << " and placement index " << placeIndex_
                                << " WaferType " << waferType_ << " Partial " << partialType_ << " WaferX " << x0
                                << " WaferY " << y0 << "\n\n";
  auto start_t = std::chrono::high_resolution_clock::now();
  std::cout << "v17 ? " << hgcons_.v17OrLess() << std::endl;
  for (int i = 0; i < nTrials_; i++) {
    double xi = (2 * r2 * static_cast<double>(rand()) / RAND_MAX) - r2;
    double yi = (2 * R2 * static_cast<double>(rand()) / RAND_MAX) - R2;
    bool goodPoint = true;
    int ug = 0;
    int vg = 0;
    if (partialType_ == 11 || partialType_ == 13 || partialType_ == 15 || partialType_ == 21 || partialType_ == 23 ||
        partialType_ == 25 || partialType_ == 0) {
      ug = 0;
      vg = 0;
    } else if (partialType_ == 12 || partialType_ == 14 || partialType_ == 16 || partialType_ == 22 ||
               partialType_ == 24) {
      ug = nCells + 1;
      vg = 2 * (nCells - 1);
    }
    std::pair<double, double> xyg = wafer2.cellUV2XY2(ug, vg, placeIndex_, waferType_);
    std::vector<std::pair<double, double> > wxy =
        HGCalWaferMask::waferXY(0, placeIndex_, waferSize_, 0.0, 0.0, 0.0, v17OrLess);
    for (unsigned int i = 0; i < (wxy.size() - 1); ++i) {
      double xp1 = wxy[i].first;
      double yp1 = wxy[i].second;
      double xp2 = wxy[i + 1].first;
      double yp2 = wxy[i + 1].second;
      if ((((xi - xp1) / (xp2 - xp1)) - ((yi - yp1) / (yp2 - yp1))) *
              (((xyg.first - xp1) / (xp2 - xp1)) - ((xyg.second - yp1) / (yp2 - yp1))) <=
          0) {
        goodPoint = false;
      }
    }
    if (goodPoint) {  //Only allowing (x, y) inside a partial wafer 11, placement index 2
      G4ThreeVector point(xi, yi, 0.0);
      std::pair<int32_t, int32_t> uv5;
      if (hgcons_.v17OrLess()) {
        uv5 = wafer.cellUVFromXY1(xi, yi, placeIndex_, waferType_, partialType_, true, false);
      } else {
        uv5 = wafer.cellUVFromXY2(xi, yi, placeIndex_, waferType_, partialType_, true, false);
      }
      if (guardRing_.exclude(point, zside, frontBack, layer_, waferU_, waferV_)) {
        guard_ring << xi << "," << yi << std::endl;
      }

      if (guardRingPartial_.exclude(point, zside, frontBack, layer_, waferU_, waferV_)) {
        guard_ring_partial << xi << "," << yi << std::endl;
      } else if (mouseBite_.exclude(point, zside, layer_, waferU_, waferV_)) {
        mouse_bite << xi << "," << yi << std::endl;
      } else {
        selected << xi << "," << yi << std::endl;
        outputFile << xi << "," << yi << "," << uv5.first << "," << uv5.second << "," << std::endl;
      }
    }
  }
  guard_ring.close();
  guard_ring_partial.close();
  mouse_bite.close();
  selected.close();
  outputFile.close();
  auto end_t = std::chrono::high_resolution_clock::now();
  auto diff_t = end_t - start_t;
  edm::LogVerbatim("HGCalGeom") << "Execution time for " << nTrials_
                                << " events = " << std::chrono::duration<double, std::milli>(diff_t).count() << " ms";
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalMouseBiteTester);
