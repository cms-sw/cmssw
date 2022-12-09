// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

class HGCalBHValidation : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalBHValidation(const edm::ParameterSet& ps);
  ~HGCalBHValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  edm::Service<TFileService> fs_;
  const std::string g4Label_, hits_;
  const edm::InputTag digis_;
  const int iSample_;
  const double threshold_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  const edm::EDGetTokenT<HGCalDigiCollection> tok_digi_;
  const int ringMax_;

  TH1D *hsimE1_, *hsimE2_, *hsimTm_;
  TH1D *hsimLn_, *hdigEn_, *hdigLn_;
  TH2D *hsimOc_, *hsi2Oc_, *hsi3Oc_;
  TH2D *hdigOc_, *hdi2Oc_, *hdi3Oc_;
};

HGCalBHValidation::HGCalBHValidation(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("ModuleLabel")),
      hits_((ps.getParameter<std::string>("HitCollection"))),
      digis_(ps.getParameter<edm::InputTag>("DigiCollection")),
      iSample_(ps.getParameter<int>("Sample")),
      threshold_(ps.getParameter<double>("Threshold")),
      tok_hits_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hits_))),
      tok_digi_(consumes<HGCalDigiCollection>(digis_)),
      ringMax_(50) {
  usesResource(TFileService::kSharedResource);

  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation::Input for SimHit:" << edm::InputTag(g4Label_, hits_)
                                      << "  Digits:" << digis_ << "  Sample: " << iSample_ << "  Threshold "
                                      << threshold_;
}

void HGCalBHValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::string>("HitCollection", "HGCHitsHEback");
  desc.add<edm::InputTag>("DigiCollection", edm::InputTag("simHGCalUnsuppressedDigis", "HEback"));
  desc.add<int>("Sample", 5);
  desc.add<double>("Threshold", 15.0);
  descriptions.add("hgcalBHAnalysis", desc);
}

void HGCalBHValidation::beginRun(edm::Run const&, edm::EventSetup const& es) {
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation::Maximum Number of"
                                      << " ring sectors:" << ringMax_ << "\nHitsValidationHcal::Booking the Histograms";

  //Histograms for Sim Hits
  hsimE1_ = fs_->make<TH1D>("SimHitEn1", "Sim Hit Energy", 1000, 0.0, 1.0);
  hsimE2_ = fs_->make<TH1D>("SimHitEn2", "Sim Hit Energy", 1000, 0.0, 1.0);
  hsimTm_ = fs_->make<TH1D>("SimHitTime", "Sim Hit Time", 1000, 0.0, 500.0);
  hsimLn_ = fs_->make<TH1D>("SimHitLong", "Sim Hit Long. Profile", 50, 0.0, 25.0);
  hsimOc_ = fs_->make<TH2D>("SimHitOccup", "Sim Hit Occupnacy", 2 * ringMax_ + 1, -ringMax_, ringMax_ + 1, 360, 0, 360);
  hsi2Oc_ = fs_->make<TH2D>("SimHitOccu2", "Sim Hit Occupnacy", 2 * ringMax_ + 1, -ringMax_, ringMax_ + 1, 360, 0, 360);
  hsi3Oc_ = fs_->make<TH2D>("SimHitOccu3", "Sim Hit Occupnacy", 2 * ringMax_ + 1, -ringMax_, ringMax_ + 1, 50, 0, 25);
  //Histograms for Digis
  hdigEn_ = fs_->make<TH1D>("DigiEnergy", "Digi ADC Sample", 1000, 0.0, 1000.0);
  hdigLn_ = fs_->make<TH1D>("DigiLong", "Digi Long. Profile", 50, 0.0, 25.0);
  hdigOc_ = fs_->make<TH2D>("DigiOccup", "Digi Occupnacy", 2 * ringMax_ + 1, -ringMax_, ringMax_ + 1, 360, 0, 360);
  hdi2Oc_ = fs_->make<TH2D>("DigiOccu2", "Digi Occupnacy", 2 * ringMax_ + 1, -ringMax_, ringMax_ + 1, 360, 0, 360);
  hdi3Oc_ = fs_->make<TH2D>("DigiOccu3", "Digi Occupnacy", 2 * ringMax_ + 1, -ringMax_, ringMax_ + 1, 50, 0, 25);
}

void HGCalBHValidation::analyze(const edm::Event& e, const edm::EventSetup&) {
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation:Run = " << e.id().run() << " Event = " << e.id().event();

  //SimHits
  const edm::Handle<edm::PCaloHitContainer>& hitsHE = e.getHandle(tok_hits_);
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation.: PCaloHitContainer"
                                      << " obtained with flag " << hitsHE.isValid();
  if (hitsHE.isValid()) {
    edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation: PCaloHit buffer " << hitsHE->size();
    unsigned i(0);
    std::map<unsigned int, double> map_try;
    for (edm::PCaloHitContainer::const_iterator it = hitsHE->begin(); it != hitsHE->end(); ++it) {
      double energy = it->energy();
      double time = it->time();
      unsigned int id = it->id();
      if (DetId(id).det() == DetId::HGCalHSc) {
        double ring = 0.01 + HGCScintillatorDetId(id).ring();
        double phi = HGCScintillatorDetId(id).iphi() - 0.01;
        int lay = HGCScintillatorDetId(id).layer();
        double ring1 = ring * HGCScintillatorDetId(id).zside();
        hsi2Oc_->Fill(ring1, phi, energy);
        hsimE1_->Fill(energy);
        hsimTm_->Fill(time, energy);
        hsimOc_->Fill(ring1, phi, energy);
        hsimLn_->Fill(lay, energy);
        hsi3Oc_->Fill(ring1, lay, energy);
        double ensum(0);
        if (map_try.count(id) != 0)
          ensum = map_try[id];
        ensum += energy;
        map_try[id] = ensum;
        edm::LogVerbatim("HGCalValidation") << "HGCalBHHit[" << ++i << "] ID " << std::hex << " " << id << std::dec
                                            << " " << HGCScintillatorDetId(id) << " E " << energy << " time " << time;
      }
    }
    for (std::map<unsigned int, double>::iterator itr = map_try.begin(); itr != map_try.end(); ++itr) {
      hsimE2_->Fill((*itr).second);
    }
  }

  //Digits
  unsigned int kount(0);
  const edm::Handle<HGCalDigiCollection>& hecoll = e.getHandle(tok_digi_);
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation.: "
                                      << "HGCalDigiCollection obtained with"
                                      << " flag " << hecoll.isValid();
  if (hecoll.isValid()) {
    edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation: HGCalDigi "
                                        << "buffer " << hecoll->size();
    for (HGCalDigiCollection::const_iterator it = hecoll->begin(); it != hecoll->end(); ++it) {
      const HGCalDataFrame& df(*it);
      double energy = df[iSample_].data();
      if (DetId(df.id()).det() == DetId::HGCalHSc) {
        HGCScintillatorDetId cell(df.id());
        int depth = cell.layer();
        if (energy > threshold_) {
          double ring = 0.01 + cell.ring();
          double phi = cell.iphi() - 0.01;
          double ring1 = cell.zside() * ring;
          hdi2Oc_->Fill(ring1, phi);
          hdigEn_->Fill(energy);
          hdigOc_->Fill(ring1, phi);
          hdigLn_->Fill(depth);
          hdi3Oc_->Fill(ring1, depth);
          edm::LogVerbatim("HGCalValidation")
              << "HGCalBHDigit[" << ++kount << "] ID " << cell << " E " << energy << ":" << (energy > threshold_);
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(HGCalBHValidation);
