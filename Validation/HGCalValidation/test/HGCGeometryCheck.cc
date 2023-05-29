#include <memory>
#include <iostream>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/transform.h"

#include "SimDataFormats/ValidationFormats/interface/PHGCalValidInfo.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <TH2.h>

//#define EDM_ML_DEBUG

class HGCGeometryCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCGeometryCheck(const edm::ParameterSet &);
  ~HGCGeometryCheck() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void beginJob() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

private:
  const edm::EDGetTokenT<PHGCalValidInfo> g4Token_;
  const std::vector<std::string> geometrySource_;
  const std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> > tok_hgcGeom_;

  //HGCal geometry scheme
  std::vector<const HGCalDDDConstants *> hgcGeometry_;

  //histogram related stuff
  TH2F *heedzVsZ, *hefdzVsZ, *hebdzVsZ;
  TH2F *heezVsLayer, *hefzVsLayer, *hebzVsLayer;
  TH2F *heerVsLayer, *hefrVsLayer, *hebrVsLayer;

  static constexpr double mmTocm_ = 0.1;
};

HGCGeometryCheck::HGCGeometryCheck(const edm::ParameterSet &cfg)
    : g4Token_(consumes<PHGCalValidInfo>(cfg.getParameter<edm::InputTag>("g4Source"))),
      geometrySource_(cfg.getUntrackedParameter<std::vector<std::string> >("geometrySource")),
      tok_hgcGeom_{edm::vector_transform(geometrySource_, [this](const std::string &name) {
        return esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name});
      })} {
  usesResource(TFileService::kSharedResource);

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
    const edm::ESHandle<HGCalDDDConstants> &hgcGeom = iSetup.getHandle(tok_hgcGeom_[i]);
    if (hgcGeom.isValid()) {
      hgcGeometry_.push_back(hgcGeom.product());
      edm::LogVerbatim("HGCalValid") << "Initialize geometry for " << geometrySource_[i];
    } else {
      edm::LogWarning("HGCalValid") << "Cannot initiate HGCalDDDConstants for " << geometrySource_[i];
    }
  }
}

void HGCGeometryCheck::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalValid") << "HGCGeometryCheck::Run " << iEvent.id().run() << " Event " << iEvent.id().event()
                                 << " Luminosity " << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing();
#endif
  //Accessing G4 information
  const edm::Handle<PHGCalValidInfo> &infoLayer = iEvent.getHandle(g4Token_);

  if (infoLayer.isValid()) {
    //step vertex information
    const std::vector<float> &hitVtxX = infoLayer->hitvtxX();
    const std::vector<float> &hitVtxY = infoLayer->hitvtxY();
    const std::vector<float> &hitVtxZ = infoLayer->hitvtxZ();
    const std::vector<unsigned int> &hitDet = infoLayer->hitDets();
    const std::vector<unsigned int> &hitIdx = infoLayer->hitIndex();

    //loop over all hits
    for (unsigned int i = 0; i < hitVtxZ.size(); i++) {
      double xx = mmTocm_ * hitVtxX[i];
      double yy = mmTocm_ * hitVtxY[i];
      double zz = mmTocm_ * hitVtxZ[i];
      double rr = sqrt(xx * xx + yy * yy);
      if ((hitDet[i] == static_cast<unsigned int>(DetId::Forward)) ||
          (hitDet[i] == static_cast<unsigned int>(DetId::HGCalEE)) ||
          (hitDet[i] == static_cast<unsigned int>(DetId::HGCalHSi)) ||
          (hitDet[i] == static_cast<unsigned int>(DetId::HGCalHSc))) {
        int dtype(0), layer(0), zside(1);
        if ((hitDet[i] == static_cast<unsigned int>(DetId::HGCalEE)) ||
            (hitDet[i] == static_cast<unsigned int>(DetId::HGCalHSi))) {
          HGCSiliconDetId id(hitIdx[i]);
          dtype = (id.det() == DetId::HGCalEE) ? 0 : 1;
          layer = id.layer();
          zside = id.zside();
        } else {
          HGCScintillatorDetId id(hitIdx[i]);
          dtype = 2;
          layer = id.layer();
          zside = id.zside();
        }
        double zp = hgcGeometry_[dtype]->waferZ(layer, true);  //cm
        if (zside < 0)
          zp = -zp;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalValid") << "Info[" << i << "] Detector Information " << hitDet[i] << ":" << zside << ":"
                                       << layer << " Z " << zp << ":" << zz << " R " << rr;
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
      }
    }  //end G4 hits
  } else {
    edm::LogWarning("HGCalValid") << "No PHGCalInfo " << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryCheck);
