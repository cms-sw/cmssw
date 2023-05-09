// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

class HGCalSimHitValidation : public DQMEDAnalyzer {
public:
  struct energysum {
    energysum() {
      etotal = 0;
      for (int i = 0; i < 6; ++i)
        eTime[i] = 0.;
    }
    double eTime[6], etotal;
  };

  struct hitsinfo {
    hitsinfo() {
      x = y = z = phi = eta = 0.0;
      cell = cell2 = sector = sector2 = type = layer = 0;
    }
    double x, y, z, phi, eta;
    int cell, cell2, sector, sector2, type, layer;
  };

  explicit HGCalSimHitValidation(const edm::ParameterSet&);
  ~HGCalSimHitValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void analyzeHits(std::vector<PCaloHit>& hits);
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  void fillHitsInfo(std::pair<hitsinfo, energysum> hit_, unsigned int itimeslice, double esum);

  TH1F* createHisto(std::string histname, const int nbins, float minIndexX, float maxIndexX, bool isLogX = true);
  void histoSetting(TH1F*& histo,
                    const char* xTitle,
                    const char* yTitle = "",
                    Color_t lineColor = kBlack,
                    Color_t markerColor = kBlack,
                    int linewidth = 1);
  void histoSetting(TH2F*& histo,
                    const char* xTitle,
                    const char* yTitle = "",
                    Color_t lineColor = kBlack,
                    Color_t markerColor = kBlack,
                    int linewidth = 1);
  void fillMuonTomoHistos(int partialType, std::pair<hitsinfo, energysum> hit_);

  // ----------member data ---------------------------
  const std::string nameDetector_, caloHitSource_;
  const HGCalDDDConstants* hgcons_;
  const std::vector<double> times_;
  const int verbosity_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> tok_hgcal_;
  const edm::EDGetTokenT<edm::HepMCProduct> tok_hepMC_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  unsigned int layers_;
  int firstLayer_;

  std::vector<MonitorElement*> HitOccupancy_Plus_, HitOccupancy_Minus_;
  std::vector<MonitorElement*> EtaPhi_Plus_, EtaPhi_Minus_;
  MonitorElement *MeanHitOccupancy_Plus_, *MeanHitOccupancy_Minus_;
  static const unsigned int maxTime_ = 6;
  std::vector<MonitorElement*> energy_[maxTime_];
  std::vector<MonitorElement*> energyFWF_, energyFWCN_, energyFWCK_;
  std::vector<MonitorElement*> energyPWF_, energyPWCN_, energyPWCK_;
  std::vector<MonitorElement*> hitXYFWF_, hitXYFWCN_, hitXYFWCK_, hitXYB_;
  unsigned int nTimes_;
};

HGCalSimHitValidation::HGCalSimHitValidation(const edm::ParameterSet& iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
      caloHitSource_(iConfig.getParameter<std::string>("CaloHitSource")),
      times_(iConfig.getParameter<std::vector<double> >("TimeSlices")),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      tok_hgcal_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      tok_hepMC_(consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"))),
      tok_hits_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", caloHitSource_))),
      firstLayer_(1) {
  nTimes_ = (times_.size() > maxTime_) ? maxTime_ : times_.size();
}

void HGCalSimHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<double> times = {25.0, 1000.0};
  desc.add<std::string>("DetectorName", "HGCalEESensitive");
  desc.add<std::string>("CaloHitSource", "HGCHitsEE");
  desc.add<std::vector<double> >("TimeSlices", times);
  desc.addUntracked<int>("Verbosity", 0);
  desc.addUntracked<bool>("TestNumber", true);
  descriptions.add("hgcalSimHitValidationEE", desc);
}

void HGCalSimHitValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Generator input
  if (verbosity_ > 0) {
    const edm::Handle<edm::HepMCProduct>& evtMC = iEvent.getHandle(tok_hepMC_);
    if (!evtMC.isValid()) {
      edm::LogVerbatim("HGCalValidation") << "no HepMCProduct found";
    } else {
      const HepMC::GenEvent* myGenEvent = evtMC->GetEvent();
      unsigned int k(0);
      for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
           ++p, ++k) {
        edm::LogVerbatim("HGCalValidation") << "Particle[" << k << "] with pt " << (*p)->momentum().perp() << " eta "
                                            << (*p)->momentum().eta() << " phi " << (*p)->momentum().phi();
      }
    }
  }

  //Now the hits
  const edm::Handle<edm::PCaloHitContainer>& theCaloHitContainers = iEvent.getHandle(tok_hits_);
  if (theCaloHitContainers.isValid()) {
    if (verbosity_ > 0)
      edm::LogVerbatim("HGCalValidation") << " PcalohitItr = " << theCaloHitContainers->size();
    std::vector<PCaloHit> caloHits;
    caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), theCaloHitContainers->end());
    analyzeHits(caloHits);
  } else if (verbosity_ > 0) {
    edm::LogVerbatim("HGCalValidation") << "PCaloHitContainer does not exist!";
  }
}

void HGCalSimHitValidation::analyzeHits(std::vector<PCaloHit>& hits) {
  std::map<int, int> OccupancyMap_plus, OccupancyMap_minus;
  OccupancyMap_plus.clear();
  OccupancyMap_minus.clear();

  std::map<uint32_t, std::pair<hitsinfo, energysum> > map_hits;
  map_hits.clear();

  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << hits.size() << " PcaloHit elements";
  unsigned int nused(0);
  for (unsigned int i = 0; i < hits.size(); i++) {
    double energy = hits[i].energy();
    double time = hits[i].time();
    uint32_t id_ = hits[i].id();
    int cell, sector, subsector(0), layer, zside;
    int cell2(0), type(0);
    if (hgcons_->waferHexagon8()) {
      HGCSiliconDetId detId = HGCSiliconDetId(id_);
      cell = detId.cellU();
      cell2 = detId.cellV();
      sector = detId.waferU();
      subsector = detId.waferV();
      type = detId.type();
      layer = detId.layer();
      zside = detId.zside();
    } else if (hgcons_->tileTrapezoid()) {
      HGCScintillatorDetId detId = HGCScintillatorDetId(id_);
      sector = detId.ietaAbs();
      cell = detId.iphi();
      subsector = 1;
      type = detId.type();
      layer = detId.layer();
      zside = detId.zside();
    } else {
      edm::LogError("HGCalValidation") << "Wrong geometry mode " << hgcons_->geomMode();
      continue;
    }
    nused++;
    if (verbosity_ > 1)
      edm::LogVerbatim("HGCalValidation")
          << "Detector " << nameDetector_ << " zside = " << zside << " sector|wafer = " << sector << ":" << subsector
          << " type = " << type << " layer = " << layer << " cell = " << cell << ":" << cell2 << " energy = " << energy
          << " energyem = " << hits[i].energyEM() << " energyhad = " << hits[i].energyHad() << " time = " << time;

    HepGeom::Point3D<float> gcoord;
    std::pair<float, float> xy;
    if (hgcons_->waferHexagon8()) {
      xy = hgcons_->locateCell(zside, layer, sector, subsector, cell, cell2, false, true, false, false);
    } else {
      xy = hgcons_->locateCellTrap(zside, layer, sector, cell, false, false);
    }
    double zp = hgcons_->waferZ(layer, false);
    if (zside < 0)
      zp = -zp;
    float xp = (zp < 0) ? -xy.first : xy.first;
    gcoord = HepGeom::Point3D<float>(xp, xy.second, zp);
    double tof = (gcoord.mag() * CLHEP::mm) / CLHEP::c_light;
    if (verbosity_ > 1)
      edm::LogVerbatim("HGCalValidation")
          << std::hex << id_ << std::dec << " global coordinate " << gcoord << " time " << time << ":" << tof;
    time -= tof;

    energysum esum;
    hitsinfo hinfo;
    if (map_hits.count(id_) != 0) {
      hinfo = map_hits[id_].first;
      esum = map_hits[id_].second;
    } else {
      hinfo.x = gcoord.x();
      hinfo.y = gcoord.y();
      hinfo.z = gcoord.z();
      hinfo.sector = sector;
      hinfo.sector2 = subsector;
      hinfo.cell = cell;
      hinfo.cell2 = cell2;
      hinfo.type = type;
      hinfo.layer = layer - firstLayer_;
      hinfo.phi = gcoord.getPhi();
      hinfo.eta = gcoord.getEta();
    }
    esum.etotal += energy;
    for (unsigned int k = 0; k < nTimes_; ++k) {
      if (time > 0 && time < times_[k])
        esum.eTime[k] += energy;
    }

    if (verbosity_ > 1)
      edm::LogVerbatim("HGCalValidation") << " -----------------------   gx = " << hinfo.x << " gy = " << hinfo.y
                                          << " gz = " << hinfo.z << " phi = " << hinfo.phi << " eta = " << hinfo.eta;
    map_hits[id_] = std::pair<hitsinfo, energysum>(hinfo, esum);
  }
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << map_hits.size()
                                        << " detector elements being hit";

  std::map<uint32_t, std::pair<hitsinfo, energysum> >::iterator itr;
  for (itr = map_hits.begin(); itr != map_hits.end(); ++itr) {
    hitsinfo hinfo = (*itr).second.first;
    energysum esum = (*itr).second.second;
    int layer = hinfo.layer;
    double eta = hinfo.eta;
    int type, part, orient;
    int partialType = -1;
    if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHESiliconSensitive")) {
      HGCSiliconDetId detId = HGCSiliconDetId((*itr).first);
      std::tie(type, part, orient) = hgcons_->waferType(detId, false);
      partialType = part;
    }

    for (unsigned int itimeslice = 0; itimeslice < nTimes_; itimeslice++) {
      fillHitsInfo((*itr).second, itimeslice, esum.eTime[itimeslice]);
    }

    if (eta > 0.0)
      fillOccupancyMap(OccupancyMap_plus, layer);
    else
      fillOccupancyMap(OccupancyMap_minus, layer);

    fillMuonTomoHistos(partialType, (*itr).second);
  }
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << "With map:used:total " << hits.size() << "|" << nused << "|"
                                        << map_hits.size() << " hits";

  for (auto const& itr : OccupancyMap_plus) {
    int layer = itr.first;
    int occupancy = itr.second;
    HitOccupancy_Plus_.at(layer)->Fill(occupancy);
  }
  for (auto const& itr : OccupancyMap_minus) {
    int layer = itr.first;
    int occupancy = itr.second;
    HitOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
}

void HGCalSimHitValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer) {
  if (OccupancyMap.find(layer) != OccupancyMap.end()) {
    ++OccupancyMap[layer];
  } else {
    OccupancyMap[layer] = 1;
  }
}

void HGCalSimHitValidation::fillHitsInfo(std::pair<hitsinfo, energysum> hits, unsigned int itimeslice, double esum) {
  unsigned int ilayer = hits.first.layer;
  if (ilayer < layers_) {
    energy_[itimeslice].at(ilayer)->Fill(esum);
    if (itimeslice == 0) {
      EtaPhi_Plus_.at(ilayer)->Fill(hits.first.eta, hits.first.phi);
      EtaPhi_Minus_.at(ilayer)->Fill(hits.first.eta, hits.first.phi);
    }
  } else {
    if (verbosity_ > 0)
      edm::LogVerbatim("HGCalValidation")
          << "Problematic Hit for " << nameDetector_ << " at sector " << hits.first.sector << ":" << hits.first.sector2
          << " layer " << hits.first.layer << " cell " << hits.first.cell << ":" << hits.first.cell2 << " energy "
          << hits.second.etotal;
  }
}

void HGCalSimHitValidation::fillMuonTomoHistos(int partialType, std::pair<hitsinfo, energysum> hits) {
  hitsinfo hinfo = hits.first;
  energysum esum = hits.second;
  double edep =
      esum.eTime[0] * CLHEP::GeV /
      CLHEP::keV;  //index 0 and 1 corresponds to 25 ns and 1000 ns, respectively. In addititon, chaging energy loss unit to keV.

  unsigned int ilayer = hinfo.layer;
  double x = hinfo.x * CLHEP::mm / CLHEP::cm;  // chaging length unit to cm.
  double y = hinfo.y * CLHEP::mm / CLHEP::cm;
  if (ilayer < layers_) {
    if (nameDetector_ == "HGCalEESensitive" or nameDetector_ == "HGCalHESiliconSensitive") {
      // Fill the energy loss histograms for MIP
      if (!TMath::AreEqualAbs(edep, 0.0, 1.e-5)) {  //to avoid peak at zero due Eloss less than 10 mili eV.
        if (hinfo.type == HGCSiliconDetId::HGCalFine) {
          if (partialType == 0)
            energyFWF_.at(ilayer)->Fill(edep);
          if (partialType > 0)
            energyPWF_.at(ilayer)->Fill(edep);
        }
        if (hinfo.type == HGCSiliconDetId::HGCalCoarseThin) {
          if (partialType == 0)
            energyFWCN_.at(ilayer)->Fill(edep);
          if (partialType > 0)
            energyPWCN_.at(ilayer)->Fill(edep);
        }
        if (hinfo.type == HGCSiliconDetId::HGCalCoarseThick) {
          if (partialType == 0)
            energyFWCK_.at(ilayer)->Fill(edep);
          if (partialType > 0)
            energyPWCK_.at(ilayer)->Fill(edep);
        }
      }

      // Fill the XY distribution of detector hits
      if (hinfo.type == HGCSiliconDetId::HGCalFine)
        hitXYFWF_.at(ilayer)->Fill(x, y);

      if (hinfo.type == HGCSiliconDetId::HGCalCoarseThin)
        hitXYFWCN_.at(ilayer)->Fill(x, y);

      if (hinfo.type == HGCSiliconDetId::HGCalCoarseThick)
        hitXYFWCK_.at(ilayer)->Fill(x, y);

    }  //is Silicon
    if (nameDetector_ == "HGCalHEScintillatorSensitive") {
      hitXYB_.at(ilayer)->Fill(x, y);
    }  //is Scintillator
  }    //layer condition
}

// ------------ method called when starting to processes a run  ------------
void HGCalSimHitValidation::dqmBeginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  hgcons_ = &iSetup.getData(tok_hgcal_);
  layers_ = hgcons_->layers(false);
  firstLayer_ = hgcons_->firstLayer();
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " defined with " << layers_ << " Layers with first at "
                                        << firstLayer_;
}

void HGCalSimHitValidation::bookHistograms(DQMStore::IBooker& iB, edm::Run const&, edm::EventSetup const&) {
  iB.setCurrentFolder("HGCAL/HGCalSimHitsV/" + nameDetector_);

  std::ostringstream histoname;
  for (unsigned int il = 0; il < layers_; ++il) {
    int ilayer = firstLayer_ + static_cast<int>(il);
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    histoname.str("");
    histoname << "HitOccupancy_Plus_layer_" << istr1;
    HitOccupancy_Plus_.push_back(iB.book1D(histoname.str().c_str(), "HitOccupancy_Plus", 501, -0.5, 500.5));
    histoname.str("");
    histoname << "HitOccupancy_Minus_layer_" << istr1;
    HitOccupancy_Minus_.push_back(iB.book1D(histoname.str().c_str(), "HitOccupancy_Minus", 501, -0.5, 500.5));

    histoname.str("");
    histoname << "EtaPhi_Plus_"
              << "layer_" << istr1;
    EtaPhi_Plus_.push_back(iB.book2D(histoname.str().c_str(), "Occupancy", 31, 1.45, 3.0, 72, -CLHEP::pi, CLHEP::pi));
    histoname.str("");
    histoname << "EtaPhi_Minus_"
              << "layer_" << istr1;
    EtaPhi_Minus_.push_back(
        iB.book2D(histoname.str().c_str(), "Occupancy", 31, -3.0, -1.45, 72, -CLHEP::pi, CLHEP::pi));

    for (unsigned int itimeslice = 0; itimeslice < nTimes_; itimeslice++) {
      histoname.str("");
      histoname << "energy_time_" << itimeslice << "_layer_" << istr1;
      energy_[itimeslice].push_back(iB.book1D(histoname.str().c_str(), "energy_", 100, 0, 0.1));
    }

    ///////////// Histograms for Energy loss in full wafers////////////
    if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHESiliconSensitive")) {
      histoname.str("");
      histoname << "energy_FullWafer_Fine_layer_" << istr1;
      TH1F* hEdepFWF = createHisto(histoname.str(), 100, 0., 400., false);
      histoSetting(hEdepFWF, "Eloss (keV)", "", kRed, kRed, 2);
      energyFWF_.push_back(iB.book1D(histoname.str().c_str(), hEdepFWF));
      hEdepFWF->Delete();

      histoname.str("");
      histoname << "energy_FullWafer_CoarseThin_layer_" << istr1;
      TH1F* hEdepFWCN = createHisto(histoname.str(), 100, 0., 400., false);
      histoSetting(hEdepFWCN, "Eloss (keV)", "", kGreen + 1, kGreen + 1, 2);
      energyFWCN_.push_back(iB.book1D(histoname.str().c_str(), hEdepFWCN));
      hEdepFWCN->Delete();

      histoname.str("");
      histoname << "energy_FullWafer_CoarseThick_layer_" << istr1;
      TH1F* hEdepFWCK = createHisto(histoname.str(), 100, 0., 400., false);
      histoSetting(hEdepFWCK, "Eloss (keV)", "", kMagenta, kMagenta, 2);
      energyFWCK_.push_back(iB.book1D(histoname.str().c_str(), hEdepFWCK));
      hEdepFWCK->Delete();
    }

    ///////////////////////////////////////////////////////////////////

    ///////////// Histograms for Energy loss in partial wafers////////////
    if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHESiliconSensitive")) {
      histoname.str("");
      histoname << "energy_PartialWafer_Fine_layer_" << istr1;
      TH1F* hEdepPWF = createHisto(histoname.str(), 100, 0., 400., false);
      histoSetting(hEdepPWF, "Eloss (keV)", "", kRed, kRed, 2);
      energyPWF_.push_back(iB.book1D(histoname.str().c_str(), hEdepPWF));
      hEdepPWF->Delete();

      histoname.str("");
      histoname << "energy_PartialWafer_CoarseThin_layer_" << istr1;
      TH1F* hEdepPWCN = createHisto(histoname.str(), 100, 0., 400., false);
      histoSetting(hEdepPWCN, "Eloss (keV)", "", kGreen + 1, kGreen + 1, 2);
      energyPWCN_.push_back(iB.book1D(histoname.str().c_str(), hEdepPWCN));
      hEdepPWCN->Delete();

      histoname.str("");
      histoname << "energy_PartialWafer_CoarseThick_layer_" << istr1;
      TH1F* hEdepPWCK = createHisto(histoname.str(), 100, 0., 400., false);
      histoSetting(hEdepPWCK, "Eloss (keV)", "", kMagenta, kMagenta, 2);
      energyPWCK_.push_back(iB.book1D(histoname.str().c_str(), hEdepPWCK));
      hEdepPWCK->Delete();
    }
    ///////////////////////////////////////////////////////////////////

    // ///////////// Histograms for the XY distribution of fired cells/scintillator tiles ///////////////
    if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHESiliconSensitive")) {
      histoname.str("");
      histoname << "hitXY_FullWafer_Fine_layer_" << istr1;
      TH2F* hitXYFWF = new TH2F(
          Form("hitXYFWF_%s", histoname.str().c_str()), histoname.str().c_str(), 100, -300., 300., 100, -300., 300.);
      histoSetting(hitXYFWF, "x (cm)", "y (cm)", kRed, kRed);
      hitXYFWF_.push_back(iB.book2D(histoname.str().c_str(), hitXYFWF));
      hitXYFWF->Delete();

      histoname.str("");
      histoname << "hitXY_FullWafer_CoarseThin_layer_" << istr1;
      TH2F* hitXYFWCN = new TH2F(
          Form("hitXYFWCN_%s", histoname.str().c_str()), histoname.str().c_str(), 100, -300., 300., 100, -300., 300.);
      histoSetting(hitXYFWCN, "x (cm)", "y (cm)", kGreen + 1, kGreen + 1);
      hitXYFWCN_.push_back(iB.book2D(histoname.str().c_str(), hitXYFWCN));
      hitXYFWCN->Delete();

      histoname.str("");
      histoname << "hitXY_FullWafer_CoarseThick_layer_" << istr1;
      TH2F* hitXYFWCK = new TH2F(
          Form("hitXYFWCK_%s", histoname.str().c_str()), histoname.str().c_str(), 100, -300., 300., 100, -300., 300.);
      histoSetting(hitXYFWCK, "x (cm)", "y (cm)", kMagenta, kMagenta);
      hitXYFWCK_.push_back(iB.book2D(histoname.str().c_str(), hitXYFWCK));
      hitXYFWCK->Delete();
    }

    if (nameDetector_ == "HGCalHEScintillatorSensitive") {
      histoname.str("");
      histoname << "hitXY_Scintillator_layer_" << istr1;
      TH2F* hitXYB = new TH2F(
          Form("hitXYB_%s", histoname.str().c_str()), histoname.str().c_str(), 100, -300., 300., 100, -300., 300.);
      histoSetting(hitXYB, "x (cm)", "y (cm)", kBlue, kBlue);
      hitXYB_.push_back(iB.book2D(histoname.str().c_str(), hitXYB));
      hitXYB->Delete();
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////
  }
  MeanHitOccupancy_Plus_ = iB.book1D("MeanHitOccupancy_Plus", "MeanHitOccupancy_Plus", layers_, 0.5, layers_ + 0.5);
  MeanHitOccupancy_Minus_ = iB.book1D("MeanHitOccupancy_Minus", "MeanHitOccupancy_Minus", layers_, 0.5, layers_ + 0.5);
}

TH1F* HGCalSimHitValidation::createHisto(
    std::string histname, const int nbins, float minIndexX, float maxIndexX, bool isLogX) {
  TH1F* hist = nullptr;
  if (isLogX) {
    Double_t xbins[nbins + 1];
    double dx = (maxIndexX - minIndexX) / nbins;
    for (int i = 0; i <= nbins; i++) {
      xbins[i] = TMath::Power(10, (minIndexX + i * dx));
    }
    hist = new TH1F(Form("hEdep_%s", histname.c_str()), histname.c_str(), nbins, xbins);
  } else {
    hist = new TH1F(Form("hEdep_%s", histname.c_str()), histname.c_str(), nbins, minIndexX, maxIndexX);
  }
  return hist;
}

void HGCalSimHitValidation::histoSetting(
    TH1F*& histo, const char* xTitle, const char* yTitle, Color_t lineColor, Color_t markerColor, int lineWidth) {
  histo->SetStats();
  histo->SetLineColor(lineColor);
  histo->SetLineWidth(lineWidth);
  histo->SetMarkerColor(markerColor);
  histo->GetXaxis()->SetTitle(xTitle);
  histo->GetYaxis()->SetTitle(yTitle);
}

void HGCalSimHitValidation::histoSetting(
    TH2F*& histo, const char* xTitle, const char* yTitle, Color_t lineColor, Color_t markerColor, int lineWidth) {
  histo->SetStats();
  histo->SetLineColor(lineColor);
  histo->SetLineWidth(lineWidth);
  histo->SetMarkerColor(markerColor);
  histo->SetMarkerStyle(kFullCircle);
  histo->SetMarkerSize(0.6);
  histo->GetXaxis()->SetTitle(xTitle);
  histo->GetYaxis()->SetTitle(yTitle);
}
#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalSimHitValidation);
