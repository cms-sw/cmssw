// system include files
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

// user include files

class HGCalDigiValidation : public DQMEDAnalyzer {
public:
  struct digiInfo {
    digiInfo() {
      x = y = z = charge = 0.0;
      layer = adc = 0;
      mode = threshold = false;
    }
    double x, y, z, charge;
    int layer, adc;
    bool mode, threshold;  //tot mode and zero supression
  };

  explicit HGCalDigiValidation(const edm::ParameterSet&);
  ~HGCalDigiValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillDigiInfo(digiInfo& hinfo);
  void fillDigiInfo();
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  template <class T1, class T2>
  void digiValidation(
      const T1& detId, const T2* geom, int layer, uint16_t adc, double charge, bool mode, bool threshold);

  // ----------member data ---------------------------
  const std::string nameDetector_;
  const bool ifNose_;
  const int verbosity_, SampleIndx_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> tok_hgcalc_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcalg_;
  const edm::EDGetTokenT<HGCalDigiCollection> digiSource_;
  int layers_, firstLayer_;

  std::map<int, int> OccupancyMap_plus_;
  std::map<int, int> OccupancyMap_minus_;

  std::vector<MonitorElement*> TOA_;
  std::vector<MonitorElement*> DigiOccupancy_XY_;
  std::vector<MonitorElement*> ADC_;
  std::vector<MonitorElement*> TOT_;
  std::vector<MonitorElement*> DigiOccupancy_Plus_;
  std::vector<MonitorElement*> DigiOccupancy_Minus_;
  MonitorElement* MeanDigiOccupancy_Plus_;
  MonitorElement* MeanDigiOccupancy_Minus_;
};

HGCalDigiValidation::HGCalDigiValidation(const edm::ParameterSet& iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
      ifNose_(iConfig.getParameter<bool>("ifNose")),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      SampleIndx_(iConfig.getUntrackedParameter<int>("SampleIndx", 0)),
      tok_hgcalc_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      tok_hgcalg_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameDetector_})),
      digiSource_(consumes<HGCalDigiCollection>(iConfig.getParameter<edm::InputTag>("DigiSource"))),
      firstLayer_(1) {}

void HGCalDigiValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("DetectorName", "HGCalEESensitive");
  desc.add<edm::InputTag>("DigiSource", edm::InputTag("hgcalDigis", "EE"));
  desc.add<bool>("ifNose", false);
  desc.addUntracked<int>("Verbosity", 0);
  desc.addUntracked<int>("SampleIndx", 2);  // central bx
  descriptions.add("hgcalDigiValidationEEDefault", desc);
}

void HGCalDigiValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  OccupancyMap_plus_.clear();
  OccupancyMap_minus_.clear();

  int geomType(0);
  const HGCalGeometry* geom0 = &iSetup.getData(tok_hgcalg_);
  if (geom0->topology().waferHexagon8())
    geomType = 1;
  else
    geomType = 2;
  if (nameDetector_ == "HGCalHFNoseSensitive")
    geomType = 3;

  unsigned int ntot(0), nused(0);
  if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHFNoseSensitive")) {
    //HGCalEE
    const edm::Handle<HGCalDigiCollection>& theHGCEEDigiContainers = iEvent.getHandle(digiSource_);
    if (theHGCEEDigiContainers.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << nameDetector_ << " with " << theHGCEEDigiContainers->size() << " element(s)";
      for (const auto& it : *(theHGCEEDigiContainers.product())) {
        ntot++;
        nused++;
        DetId detId = it.id();
        int layer = ((geomType == 1) ? HGCSiliconDetId(detId).layer() : HFNoseDetId(detId).layer());
        const HGCSample& hgcSample = it.sample(SampleIndx_);
        uint16_t gain = hgcSample.toa();
        uint16_t adc = hgcSample.data();
        double charge = gain;
        bool totmode = hgcSample.mode();
        bool zerothreshold = hgcSample.threshold();
        digiValidation(detId, geom0, layer, adc, charge, totmode, zerothreshold);
      }
      fillDigiInfo();
    } else {
      edm::LogVerbatim("HGCalValidation") << "DigiCollection handle does not "
                                          << "exist for " << nameDetector_;
    }
  } else if ((nameDetector_ == "HGCalHESiliconSensitive") || (nameDetector_ == "HGCalHEScintillatorSensitive")) {
    //HGCalHE
    const edm::Handle<HGCalDigiCollection>& theHGCHEDigiContainers = iEvent.getHandle(digiSource_);
    if (theHGCHEDigiContainers.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << nameDetector_ << " with " << theHGCHEDigiContainers->size() << " element(s)";
      for (const auto& it : *(theHGCHEDigiContainers.product())) {
        ntot++;
        nused++;
        DetId detId = it.id();
        int layer = ((geomType == 1) ? HGCSiliconDetId(detId).layer() : HGCScintillatorDetId(detId).layer());
        const HGCSample& hgcSample = it.sample(SampleIndx_);
        uint16_t gain = hgcSample.toa();
        uint16_t adc = hgcSample.data();
        double charge = gain;
        bool totmode = hgcSample.mode();
        bool zerothreshold = hgcSample.threshold();
        digiValidation(detId, geom0, layer, adc, charge, totmode, zerothreshold);
      }
      fillDigiInfo();
    } else {
      edm::LogVerbatim("HGCalValidation") << "DigiCollection handle does not "
                                          << "exist for " << nameDetector_;
    }
  } else {
    edm::LogWarning("HGCalValidation") << "invalid detector name !! " << nameDetector_;
  }
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << "Event " << iEvent.id().event() << " with " << ntot << " total and " << nused
                                        << " used digis";
}

template <class T1, class T2>
void HGCalDigiValidation::digiValidation(
    const T1& detId, const T2* geom, int layer, uint16_t adc, double charge, bool mode, bool threshold) {
  if (verbosity_ > 1)
    edm::LogVerbatim("HGCalValidation") << std::hex << detId.rawId() << std::dec << " " << detId.rawId();
  DetId id1 = DetId(detId.rawId());
  const GlobalPoint& global1 = geom->getPosition(id1);

  if (verbosity_ > 1)
    edm::LogVerbatim("HGCalValidation") << " adc = " << adc << " toa = " << charge;

  digiInfo hinfo;
  hinfo.x = global1.x();
  hinfo.y = global1.y();
  hinfo.z = global1.z();
  hinfo.adc = adc;
  hinfo.charge = charge;
  hinfo.layer = layer - firstLayer_;
  hinfo.mode = mode;
  hinfo.threshold = threshold;

  if (verbosity_ > 1)
    edm::LogVerbatim("HGCalValidation") << "gx =  " << hinfo.x << " gy = " << hinfo.y << " gz = " << hinfo.z;

  if (global1.eta() > 0)
    fillOccupancyMap(OccupancyMap_plus_, hinfo.layer);
  else
    fillOccupancyMap(OccupancyMap_minus_, hinfo.layer);

  fillDigiInfo(hinfo);
}

void HGCalDigiValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer) {
  if (OccupancyMap.find(layer) != OccupancyMap.end())
    OccupancyMap[layer]++;
  else
    OccupancyMap[layer] = 1;
}

void HGCalDigiValidation::fillDigiInfo(digiInfo& hinfo) {
  int ilayer = hinfo.layer;
  TOA_.at(ilayer)->Fill(hinfo.charge);

  if (hinfo.mode) {
    TOT_.at(ilayer)->Fill(hinfo.adc);
  }

  if (!hinfo.mode && hinfo.threshold) {
    ADC_.at(ilayer)->Fill(hinfo.adc);
    DigiOccupancy_XY_.at(ilayer)->Fill(hinfo.x, hinfo.y);
  }
}

void HGCalDigiValidation::fillDigiInfo() {
  for (const auto& itr : OccupancyMap_plus_) {
    int layer = itr.first;
    int occupancy = itr.second;
    DigiOccupancy_Plus_.at(layer)->Fill(occupancy);
  }
  for (const auto& itr : OccupancyMap_minus_) {
    int layer = itr.first;
    int occupancy = itr.second;
    DigiOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
}

void HGCalDigiValidation::dqmBeginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants* hgcons = &iSetup.getData(tok_hgcalc_);
  layers_ = hgcons->layers(true);
  firstLayer_ = hgcons->firstLayer();

  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << "current DQM directory:  "
                                        << "HGCAL/HGCalDigisV/" << nameDetector_ << "  layer = " << layers_
                                        << " with the first one at " << firstLayer_;
}

void HGCalDigiValidation::bookHistograms(DQMStore::IBooker& iB, edm::Run const&, edm::EventSetup const&) {
  iB.setCurrentFolder("HGCAL/HGCalDigisV/" + nameDetector_);

  std::ostringstream histoname;
  for (int il = 0; il < layers_; ++il) {
    int ilayer = firstLayer_ + il;
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    histoname.str("");
    histoname << "TOA_"
              << "layer_" << istr1;
    TOA_.push_back(iB.book1D(histoname.str().c_str(), "toa_", 1024, 0, 1024));

    histoname.str("");
    histoname << "ADC_"
              << "layer_" << istr1;
    ADC_.push_back(iB.book1D(histoname.str().c_str(), "ADCDigiOccupancy", 1024, 0, 1024));

    histoname.str("");
    histoname << "TOT_"
              << "layer_" << istr1;
    TOT_.push_back(iB.book1D(histoname.str().c_str(), "TOTDigiOccupancy", 4096, 0, 4096));

    histoname.str("");
    histoname << "DigiOccupancy_XY_"
              << "layer_" << istr1;
    DigiOccupancy_XY_.push_back(iB.book2D(histoname.str().c_str(), "DigiOccupancy", 50, -500, 500, 50, -500, 500));

    histoname.str("");
    histoname << "DigiOccupancy_Plus_"
              << "layer_" << istr1;
    DigiOccupancy_Plus_.push_back(iB.book1D(histoname.str().c_str(), "DigiOccupancy +z", 100, 0, 1000));
    histoname.str("");
    histoname << "DigiOccupancy_Minus_"
              << "layer_" << istr1;
    DigiOccupancy_Minus_.push_back(iB.book1D(histoname.str().c_str(), "DigiOccupancy -z", 100, 0, 1000));
  }

  histoname.str("");
  histoname << "SUMOfDigiOccupancy_Plus";
  MeanDigiOccupancy_Plus_ = iB.book1D(histoname.str().c_str(), "SUMOfDigiOccupancy_Plus", layers_, -0.5, layers_ - 0.5);
  histoname.str("");
  histoname << "SUMOfRecDigiOccupancy_Minus";
  MeanDigiOccupancy_Minus_ =
      iB.book1D(histoname.str().c_str(), "SUMOfDigiOccupancy_Minus", layers_, -0.5, layers_ - 0.5);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiValidation);
