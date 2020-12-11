#include <memory>
#include <iostream>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/ValidationFormats/interface/PHGCalValidInfo.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <TH2.h>

//#define EDM_ML_DEBUG

class HGCGeometryCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCGeometryCheck(const edm::ParameterSet &);
  ~HGCGeometryCheck() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void beginJob() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

private:
  edm::EDGetTokenT<PHGCalValidInfo> g4Token_;
  std::vector<std::string> geometrySource_;
  edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> tok_hrndc_;
  std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> > tok_hgcGeom_;

  //HGCal geometry scheme
  std::vector<const HGCalDDDConstants *> hgcGeometry_;
  const HcalDDDSimConstants *hcons_;

  //histogram related stuff
  TH2F *heedzVsZ, *hefdzVsZ, *hebdzVsZ;
  TH2F *heezVsLayer, *hefzVsLayer, *hebzVsLayer;
  TH2F *heerVsLayer, *hefrVsLayer, *hebrVsLayer;

  static constexpr double mmTocm_ = 0.1;
};

HGCGeometryCheck::HGCGeometryCheck(const edm::ParameterSet &cfg) : hcons_(nullptr) {
  usesResource(TFileService::kSharedResource);

  g4Token_ = consumes<PHGCalValidInfo>(cfg.getParameter<edm::InputTag>("g4Source"));
  geometrySource_ = cfg.getUntrackedParameter<std::vector<std::string> >("geometrySource");
  tok_hrndc_ = esConsumes<HcalDDDSimConstants, HcalSimNumberingRecord, edm::Transition::BeginRun>();
  for (const auto &name : geometrySource_) {
    if (name == "HCAL")
      tok_hgcGeom_.emplace_back(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", "HGCalHEScintillatorSensitive"}));
    else
      tok_hgcGeom_.emplace_back(
          esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name}));
  }

  edm::LogVerbatim("HGCalValid") << "HGCGeometryCheck:: use information from "
                                 << cfg.getParameter<edm::InputTag>("g4Source") << " and " << geometrySource_.size()
                                 << " geometry records:";
  for (unsigned int k = 0; k < geometrySource_.size(); ++k)
    edm::LogVerbatim("HGCalValid") << "[ " << k << "] " << geometrySource_[k];
}

void HGCGeometryCheck::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string> >("geometrySource", names);
  desc.add<edm::InputTag>("g4Source", edm::InputTag("g4SimHits", "HGCalInfoLayer"));
  descriptions.add("hgcGeomCheck", desc);
}

void HGCGeometryCheck::beginJob() {
  //initiating fileservice
  edm::Service<TFileService> fs;

  //initiating histograms
  heedzVsZ = fs->make<TH2F>("heedzVsZ", "", 1400, 315, 385, 100, -1, 1);
  hefdzVsZ = fs->make<TH2F>("hefdzVsZ", "", 2000, 350, 550, 100, -1, 1);
  hebdzVsZ = fs->make<TH2F>("hebdzVsZ", "", 360, 380, 560, 100, -5, 5);

  heezVsLayer = fs->make<TH2F>("heezVsLayer", "", 100, 0, 100, 1400, 315, 385);
  hefzVsLayer = fs->make<TH2F>("hefzVsLayer", "", 40, 0, 40, 2000, 350, 550);
  hebzVsLayer = fs->make<TH2F>("hebzVsLayer", "", 50, 0, 25, 360, 380, 560);

  heerVsLayer = fs->make<TH2F>("heerVsLayer", "", 100, 0, 100, 600, 0, 300);
  hefrVsLayer = fs->make<TH2F>("hefrVsLayer", "", 40, 0, 40, 600, 0, 300);
  hebrVsLayer = fs->make<TH2F>("hebrVsLayer", "", 50, 0, 25, 600, 0, 300);
}

void HGCGeometryCheck::beginRun(const edm::Run &, const edm::EventSetup &iSetup) {
  //initiating hgc geometry
  for (size_t i = 0; i < geometrySource_.size(); i++) {
    if (geometrySource_[i].find("Hcal") != std::string::npos) {
      edm::ESHandle<HcalDDDSimConstants> pHRNDC = iSetup.getHandle(tok_hrndc_);
      if (pHRNDC.isValid()) {
        hcons_ = &(*pHRNDC);
        hgcGeometry_.push_back(nullptr);
        edm::LogVerbatim("HGCalValid") << "Initialize geometry for " << geometrySource_[i];
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HcalGeometry for " << geometrySource_[i];
      }
    } else {
      edm::ESHandle<HGCalDDDConstants> hgcGeom = iSetup.getHandle(tok_hgcGeom_[i]);
      if (hgcGeom.isValid()) {
        hgcGeometry_.push_back(hgcGeom.product());
        edm::LogVerbatim("HGCalValid") << "Initialize geometry for " << geometrySource_[i];
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HGCalGeometry for " << geometrySource_[i];
      }
    }
  }
}

void HGCGeometryCheck::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalValid") << "HGCGeometryCheck::Run " << iEvent.id().run() << " Event " << iEvent.id().event()
                                 << " Luminosity " << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing();
#endif
  //Accessing G4 information
  edm::Handle<PHGCalValidInfo> infoLayer;
  iEvent.getByToken(g4Token_, infoLayer);

  if (infoLayer.isValid()) {
    //step vertex information
    std::vector<float> hitVtxX = infoLayer->hitvtxX();
    std::vector<float> hitVtxY = infoLayer->hitvtxY();
    std::vector<float> hitVtxZ = infoLayer->hitvtxZ();
    std::vector<unsigned int> hitDet = infoLayer->hitDets();
    std::vector<unsigned int> hitIdx = infoLayer->hitIndex();

    //loop over all hits
    for (unsigned int i = 0; i < hitVtxZ.size(); i++) {
      double xx = mmTocm_ * hitVtxX.at(i);
      double yy = mmTocm_ * hitVtxY.at(i);
      double zz = mmTocm_ * hitVtxZ.at(i);
      double rr = sqrt(xx * xx + yy * yy);
      if ((hitDet.at(i) == (unsigned int)(DetId::Forward)) || (hitDet.at(i) == (unsigned int)(DetId::HGCalEE)) ||
          (hitDet.at(i) == (unsigned int)(DetId::HGCalHSi)) || (hitDet.at(i) == (unsigned int)(DetId::HGCalHSc))) {
        int dtype(0), subdet(0), layer(0), zside(1);
        if (hitDet.at(i) == (unsigned int)(DetId::Forward)) {
          int wafer, celltype, cell;
          HGCalTestNumbering::unpackHexagonIndex(hitIdx.at(i), subdet, zside, layer, wafer, celltype, cell);
          dtype = (subdet == (int)(HGCEE)) ? 0 : 1;
        } else if ((hitDet.at(i) == (unsigned int)(DetId::HGCalEE)) ||
                   (hitDet.at(i) == (unsigned int)(DetId::HGCalHSi))) {
          HGCSiliconDetId id(hitIdx.at(i));
          dtype = (id.det() == DetId::HGCalEE) ? 0 : 1;
          layer = id.layer();
          zside = id.zside();
        } else {
          HGCScintillatorDetId id(hitIdx.at(i));
          dtype = 2;
          layer = id.layer();
          zside = id.zside();
        }
        double zp = hgcGeometry_[dtype]->waferZ(layer, true);  //cm
        if (zside < 0)
          zp = -zp;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalValid") << "Info[" << i << "] Detector Information " << hitDet[i] << ":" << subdet << ":"
                                       << zside << ":" << layer << " Z " << zp << ":" << zz << " R " << rr;
#endif
        if (dtype == 0) {
          heedzVsZ->Fill(zp, (zz - zp));
          heezVsLayer->Fill(layer, zz);
          heerVsLayer->Fill(layer, rr);
        } else if (dtype == 1) {
          hefdzVsZ->Fill(zp, (zz - zp));
          hefzVsLayer->Fill(layer, zz);
          hefrVsLayer->Fill(layer, rr);
        } else {
          hebdzVsZ->Fill(zp, (zz - zp));
          hebzVsLayer->Fill(layer, zz);
          hebrVsLayer->Fill(layer, rr);
        }

      } else if (hitDet.at(i) == (unsigned int)(DetId::Hcal)) {
        int subdet, zside, depth, eta, phi, lay;
        HcalTestNumbering::unpackHcalIndex(hitIdx.at(i), subdet, zside, depth, eta, phi, lay);
        HcalCellType::HcalCell cell = hcons_->cell(subdet, zside, lay, eta, phi);
        double zp = cell.rz / 10;  //mm --> cm
        if (zside == 0)
          zp = -zp;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalValid") << "Info[" << i << "] Detector Information " << hitDet[i] << ":" << subdet << ":"
                                       << zside << ":" << depth << ":" << eta << ":" << phi << ":" << lay << " z " << zp
                                       << ":" << zz << " R " << rr;
#endif
        hebdzVsZ->Fill(zp, (zz - zp));
        hebzVsLayer->Fill(lay, zz);
        hebrVsLayer->Fill(lay, rr);
      }

    }  //end G4 hits

  } else {
    edm::LogWarning("HGCalValid") << "No PHGCalInfo " << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryCheck);
