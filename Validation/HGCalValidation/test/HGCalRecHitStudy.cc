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
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "TH1D.h"
#include "TH2D.h"

class HGCalRecHitStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalRecHitStudy(const edm::ParameterSet&);
  ~HGCalRecHitStudy() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct energysum {
    energysum() { e15 = e25 = e50 = e100 = e250 = e1000 = 0.0; }
    double e15, e25, e50, e100, e250, e1000;
  };

  struct HitsInfo {
    HitsInfo() {
      x = y = z = time = energy = phi = eta = 0.0;
      layer = 0;
    }
    float x, y, z, time, energy, phi, eta;
    int layer;
  };

  virtual void beginJob() override {}
  virtual void endJob() override {}
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;

  template <class T1, class T2>
  void recHitValidation(DetId& detId, int layer, const T1* geom, T2 it);
  void fillHitsInfo();
  void fillHitsInfo(HitsInfo& hits);
  void fillOccupancyMap(std::map<int, int>& occupancyMap, int layer);

  // ----------member data ---------------------------
  const std::string nameDetector_;
  edm::EDGetToken recHitSource_;
  const bool ifNose_, ifHCAL_, ifLayer_;
  const int verbosity_, nbinR_, nbinZ_, nbinEta_, nLayers_;
  const double rmin_, rmax_, zmin_, zmax_, etamin_, etamax_;
  unsigned int layers_;
  int firstLayer_;
  std::map<int, int> occupancyMapPlus_, occupancyMapMinus_;

  std::vector<TH2D*> h_EtaPhiPlus_, h_EtaPhiMinus_, h_XY_;
  std::vector<TH1D*> h_energy_, h_HitOccupancyPlus_, h_HitOccupancyMinus_;
  TH1D *h_MeanHitOccupancyPlus_, *h_MeanHitOccupancyMinus_;
  TH2D *h_RZ_, *h_EtaPhi_;
};

HGCalRecHitStudy::HGCalRecHitStudy(const edm::ParameterSet& iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("detectorName")),
      ifNose_(iConfig.getUntrackedParameter<bool>("ifNose", false)),
      ifHCAL_(iConfig.getUntrackedParameter<bool>("ifHCAL", false)),
      ifLayer_(iConfig.getUntrackedParameter<bool>("ifLayer", false)),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      nbinR_(iConfig.getUntrackedParameter<int>("nBinR", 300)),
      nbinZ_(iConfig.getUntrackedParameter<int>("nBinZ", 300)),
      nbinEta_(iConfig.getUntrackedParameter<int>("nBinEta", 100)),
      nLayers_(iConfig.getUntrackedParameter<int>("layers", 28)),
      rmin_(iConfig.getUntrackedParameter<double>("rMin", 0.0)),
      rmax_(iConfig.getUntrackedParameter<double>("rMax", 300.0)),
      zmin_(iConfig.getUntrackedParameter<double>("zMin", 300.0)),
      zmax_(iConfig.getUntrackedParameter<double>("zMax", 600.0)),
      etamin_(iConfig.getUntrackedParameter<double>("etaMin", 1.2)),
      etamax_(iConfig.getUntrackedParameter<double>("etaMax", 3.2)),
      layers_(0),
      firstLayer_(1) {
  usesResource(TFileService::kSharedResource);

  auto temp = iConfig.getParameter<edm::InputTag>("source");
  if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHESiliconSensitive") ||
      (nameDetector_ == "HGCalHEScintillatorSensitive") || (nameDetector_ == "HGCalHFNoseSensitive")) {
    recHitSource_ = consumes<HGCRecHitCollection>(temp);
  } else if (nameDetector_ == "HCal") {
    if (ifHCAL_)
      recHitSource_ = consumes<HBHERecHitCollection>(temp);
    else
      recHitSource_ = consumes<HGCRecHitCollection>(temp);
  } else {
    throw cms::Exception("BadHGCRecHitSource") << "HGCal DetectorName given as " << nameDetector_ << " must be: "
                                               << "\"HGCalHESiliconSensitive\", \"HGCalHESiliconSensitive\", "
                                               << "\"HGCalHEScintillatorSensitive\", or \"HCal\"!";
  }
  edm::LogVerbatim("HGCalValidation") << "Initialize HGCalRecHitStudy for " << nameDetector_ << " with i/p tag " << temp
                                      << " Flag " << ifHCAL_ << ":" << ifNose_ << ":" << verbosity_;
}

void HGCalRecHitStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detectorName", "HGCalEESensitive");
  desc.add<edm::InputTag>("source", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.addUntracked<bool>("ifNose", false);
  desc.addUntracked<bool>("ifHCAL", false);
  desc.addUntracked<bool>("ifLayer", false);
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<int>("nBinR", 300);
  desc.addUntracked<int>("nBinZ", 300);
  desc.addUntracked<int>("nBinEta", 200);
  desc.addUntracked<int>("layers", 28);
  desc.addUntracked<double>("rMin", 0.0);
  desc.addUntracked<double>("rMax", 300.0);
  desc.addUntracked<double>("zMin", 300.0);
  desc.addUntracked<double>("zMax", 600.0);
  desc.addUntracked<double>("etaMin", 1.0);
  desc.addUntracked<double>("etaMax", 3.0);
  descriptions.add("hgcalRecHitStudyEE", desc);
}

void HGCalRecHitStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  occupancyMapPlus_.clear();
  occupancyMapMinus_.clear();

  bool ok(true);
  unsigned int ntot(0), nused(0);
  if (nameDetector_ == "HCal") {
    edm::ESHandle<CaloGeometry> geom;
    iSetup.get<CaloGeometryRecord>().get(geom);
    if (!geom.isValid())
      edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    const CaloGeometry* geom0 = geom.product();

    if (ifHCAL_) {
      edm::Handle<HBHERecHitCollection> hbhecoll;
      iEvent.getByToken(recHitSource_, hbhecoll);
      if (hbhecoll.isValid()) {
        if (verbosity_ > 0)
          edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << hbhecoll->size() << " element(s)";
        for (const auto& it : *(hbhecoll.product())) {
          DetId detId = it.id();
          ntot++;
          if (detId.subdetId() == static_cast<int>(HcalEndcap)) {
            nused++;
            int layer = HcalDetId(detId).depth();
            recHitValidation(detId, layer, geom0, &it);
          }
        }
      } else {
        ok = false;
        edm::LogWarning("HGCalValidation") << "HBHERecHitCollection Handle does not exist !!!";
      }
    } else {
      edm::Handle<HGCRecHitCollection> hbhecoll;
      iEvent.getByToken(recHitSource_, hbhecoll);
      if (hbhecoll.isValid()) {
        if (verbosity_ > 0)
          edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << hbhecoll->size() << " element(s)";
        for (const auto& it : *(hbhecoll.product())) {
          DetId detId = it.id();
          ntot++;
          nused++;
          int layer = HcalDetId(detId).depth();
          recHitValidation(detId, layer, geom0, &it);
        }
      } else {
        ok = false;
        edm::LogWarning("HGCalValidation") << "HGChebRecHitCollection Handle does not exist !!!";
      }
    }
  } else {
    edm::ESHandle<HGCalGeometry> geom;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, geom);
    if (!geom.isValid())
      edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    const HGCalGeometry* geom0 = geom.product();

    edm::Handle<HGCRecHitCollection> theRecHitContainers;
    iEvent.getByToken(recHitSource_, theRecHitContainers);
    if (theRecHitContainers.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << nameDetector_ << " with " << theRecHitContainers->size() << " element(s)";
      for (const auto& it : *(theRecHitContainers.product())) {
        ntot++;
        nused++;
        DetId detId = it.id();
        int layer = (ifNose_ ? HFNoseDetId(detId).layer()
                             : ((detId.det() == DetId::Forward)
                                    ? HGCalDetId(detId).layer()
                                    : ((detId.det() == DetId::HGCalHSc) ? HGCScintillatorDetId(detId).layer()
                                                                        : HGCSiliconDetId(detId).layer())));
        recHitValidation(detId, layer, geom0, &it);
      }
    } else {
      ok = false;
      edm::LogWarning("HGCalValidation") << "HGCRecHitCollection Handle does not exist !!!";
    }
  }
  if (ok)
    fillHitsInfo();
  edm::LogVerbatim("HGCalValidation") << "Event " << iEvent.id().event() << " with " << ntot << " total and " << nused
                                      << " used recHits";
}

template <class T1, class T2>
void HGCalRecHitStudy::recHitValidation(DetId& detId, int layer, const T1* geom, T2 it) {
  GlobalPoint global = geom->getPosition(detId);
  double energy = it->energy();

  float globalx = global.x();
  float globaly = global.y();
  float globalz = global.z();
  h_RZ_->Fill(std::abs(globalz), global.perp());
  if (ifLayer_) {
    if ((layer - firstLayer_) <= static_cast<int>(h_XY_.size()))
      h_XY_[layer - firstLayer_]->Fill(std::abs(globalx), std::abs(globaly));
  } else {
    h_EtaPhi_->Fill(std::abs(global.eta()), global.phi());
  }
  HitsInfo hinfo;
  hinfo.energy = energy;
  hinfo.x = globalx;
  hinfo.y = globaly;
  hinfo.z = globalz;
  hinfo.layer = layer - firstLayer_;
  hinfo.phi = global.phi();
  hinfo.eta = global.eta();

  if (verbosity_ > 1)
    edm::LogVerbatim("HGCalValidation") << " --------------------------   gx = " << globalx << " gy = " << globaly
                                        << " gz = " << globalz << " phi = " << hinfo.phi << " eta = " << hinfo.eta;

  fillHitsInfo(hinfo);

  if (hinfo.eta > 0)
    fillOccupancyMap(occupancyMapPlus_, hinfo.layer);
  else
    fillOccupancyMap(occupancyMapMinus_, hinfo.layer);
}

void HGCalRecHitStudy::fillOccupancyMap(std::map<int, int>& occupancyMap, int layer) {
  if (occupancyMap.find(layer) != occupancyMap.end())
    occupancyMap[layer]++;
  else
    occupancyMap[layer] = 1;
}

void HGCalRecHitStudy::fillHitsInfo() {
  for (auto const& itr : occupancyMapPlus_) {
    int layer = itr.first;
    int occupancy = itr.second;
    h_HitOccupancyPlus_.at(layer)->Fill(occupancy);
  }

  for (auto const& itr : occupancyMapMinus_) {
    int layer = itr.first;
    int occupancy = itr.second;
    h_HitOccupancyMinus_.at(layer)->Fill(occupancy);
  }
}

void HGCalRecHitStudy::fillHitsInfo(HitsInfo& hits) {
  unsigned int ilayer = hits.layer;
  h_energy_.at(ilayer)->Fill(hits.energy);
  if (!ifLayer_) {
    h_EtaPhiPlus_.at(ilayer)->Fill(hits.eta, hits.phi);
    h_EtaPhiMinus_.at(ilayer)->Fill(hits.eta, hits.phi);
  }
}

void HGCalRecHitStudy::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {
  if (nameDetector_ == "HCal") {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
    const HcalDDDRecConstants* hcons = &(*pHRNDC);
    layers_ = hcons->getMaxDepth(1);
  } else {
    edm::ESHandle<HGCalDDDConstants> pHGDC;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, pHGDC);
    const HGCalDDDConstants& hgcons_ = (*pHGDC);
    layers_ = hgcons_.layers(true);
    firstLayer_ = hgcons_.firstLayer();
  }

  edm::LogVerbatim("HGCalValidation") << "Finds " << layers_ << " layers for " << nameDetector_;

  edm::Service<TFileService> fs;
  char histoname[100], title[100];
  for (unsigned int il = 0; il < layers_; il++) {
    int ilayer = firstLayer_ + static_cast<int>(il);
    sprintf(histoname, "HitOccupancy_Plus_layer_%d", ilayer);
    h_HitOccupancyPlus_.push_back(fs->make<TH1D>(histoname, "RecHitOccupancy_Plus", 100, 0, 10000));
    sprintf(histoname, "HitOccupancy_Minus_layer_%d", ilayer);
    h_HitOccupancyMinus_.push_back(fs->make<TH1D>(histoname, "RecHitOccupancy_Minus", 100, 0, 10000));

    if (ifLayer_) {
      sprintf(histoname, "XY_L%d", ilayer);
      sprintf(title, "Y vs X at layer %d of %s", (ilayer + 1), nameDetector_.c_str());
      h_XY_.emplace_back(fs->make<TH2D>(histoname, title, nbinR_, 0, rmax_, nbinR_, 0, rmax_));
    } else {
      sprintf(histoname, "EtaPhi_Plus_Layer_%d", ilayer);
      h_EtaPhiPlus_.push_back(fs->make<TH2D>(histoname, "Occupancy", nbinEta_, etamin_, etamax_, 72, -M_PI, M_PI));
      sprintf(histoname, "EtaPhi_Minus_Layer_%d", ilayer);
      h_EtaPhiMinus_.push_back(fs->make<TH2D>(histoname, "Occupancy", nbinEta_, -etamax_, -etamin_, 72, -M_PI, M_PI));
    }

    sprintf(histoname, "Energy_Layer_%d", ilayer);
    h_energy_.push_back(fs->make<TH1D>(histoname, "Energy", 1000, 0, 10.0));
  }  //loop over layers ends here

  h_MeanHitOccupancyPlus_ =
      fs->make<TH1D>("SUMOfRecHitOccupancy_Plus", "SUMOfRecHitOccupancy_Plus", layers_, -0.5, layers_ - 0.5);
  h_MeanHitOccupancyMinus_ =
      fs->make<TH1D>("SUMOfRecHitOccupancy_Minus", "SUMOfRecHitOccupancy_Minus", layers_, -0.5, layers_ - 0.5);

  sprintf(histoname, "RZ_%s", nameDetector_.c_str());
  sprintf(title, "R vs Z for %s", nameDetector_.c_str());
  h_RZ_ = fs->make<TH2D>(histoname, title, nbinZ_, zmin_, zmax_, nbinR_, rmin_, rmax_);
  if (!ifLayer_) {
    sprintf(histoname, "EtaPhi_%s", nameDetector_.c_str());
    sprintf(title, "#phi vs #eta for %s", nameDetector_.c_str());
    h_EtaPhi_ = fs->make<TH2D>(histoname, title, nbinEta_, etamin_, etamax_, 72, -M_PI, M_PI);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalRecHitStudy);
