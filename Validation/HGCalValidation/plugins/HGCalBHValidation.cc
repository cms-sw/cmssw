// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
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
  ~HGCalBHValidation() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  virtual void beginJob() override {}
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  template <class T>
  void analyzeDigi(const T&, double const&, bool const&, int const&, unsigned int&);

private:
  edm::Service<TFileService> fs_;
  const std::string g4Label_, hcalHits_;
  const edm::InputTag hcalDigis_;
  const int iSample_, geomType_;
  const double threshold_;
  const bool ifHCAL_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  edm::EDGetToken tok_hbhe_;
  int etaMax_;

  TH1D *hsimE1_, *hsimE2_, *hsimTm_;
  TH1D *hsimLn_, *hdigEn_, *hdigLn_;
  TH2D *hsimOc_, *hsi2Oc_, *hsi3Oc_;
  TH2D *hdigOc_, *hdi2Oc_, *hdi3Oc_;
};

HGCalBHValidation::HGCalBHValidation(const edm::ParameterSet& ps)
    : g4Label_(ps.getUntrackedParameter<std::string>("ModuleLabel", "g4SimHits")),
      hcalHits_((ps.getUntrackedParameter<std::string>("HitCollection", "HcalHits"))),
      hcalDigis_(ps.getUntrackedParameter<edm::InputTag>("DigiCollection")),
      iSample_(ps.getUntrackedParameter<int>("Sample", 5)),
      geomType_(ps.getUntrackedParameter<int>("GeometryType", 0)),
      threshold_(ps.getUntrackedParameter<double>("Threshold", 12.0)),
      ifHCAL_(ps.getUntrackedParameter<bool>("ifHCAL", false)),
      etaMax_(100) {
  usesResource(TFileService::kSharedResource);

  tok_hits_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hcalHits_));
  if (ifHCAL_)
    tok_hbhe_ = consumes<QIE11DigiCollection>(hcalDigis_);
  else
    tok_hbhe_ = consumes<HGCalDigiCollection>(hcalDigis_);
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation::Input for SimHit:" << edm::InputTag(g4Label_, hcalHits_)
                                      << "  Digits:" << hcalDigis_ << "  Sample: " << iSample_ << "  Threshold "
                                      << threshold_;
}

void HGCalBHValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("ModuleLabel", "g4SimHits");
  desc.addUntracked<std::string>("HitCollection", "HcalHits");
  desc.addUntracked<edm::InputTag>("DigiCollection", edm::InputTag("hgcalDigis", "HEback"));
  desc.addUntracked<int>("Sample", 5);
  desc.addUntracked<int>("GeometryType", 0);
  desc.addUntracked<double>("Threshold", 15.0);
  desc.addUntracked<bool>("ifHCAL", false);
  descriptions.add("hgcalBHAnalysis", desc);
}

void HGCalBHValidation::beginRun(edm::Run const&, edm::EventSetup const& es) {
  if (geomType_ == 0) {
    std::string label;
    edm::ESHandle<HcalParameters> parHandle;
    es.get<HcalParametersRcd>().get(label, parHandle);
    const HcalParameters* hpar = &(*parHandle);
    const std::vector<int> etaM = hpar->etaMax;
    etaMax_ = etaM[1];
  }
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation::Maximum Number of"
                                      << " eta sectors:" << etaMax_ << "\nHitsValidationHcal::Booking the Histograms";

  //Histograms for Sim Hits
  hsimE1_ = fs_->make<TH1D>("SimHitEn1", "Sim Hit Energy", 1000, 0.0, 1.0);
  hsimE2_ = fs_->make<TH1D>("SimHitEn2", "Sim Hit Energy", 1000, 0.0, 1.0);
  hsimTm_ = fs_->make<TH1D>("SimHitTime", "Sim Hit Time", 1000, 0.0, 500.0);
  hsimLn_ = fs_->make<TH1D>("SimHitLong", "Sim Hit Long. Profile", 40, 0.0, 20.0);
  hsimOc_ = fs_->make<TH2D>("SimHitOccup", "Sim Hit Occupnacy", 2 * etaMax_ + 1, -etaMax_, etaMax_ + 1, 360, 0, 360);
  hsi2Oc_ = fs_->make<TH2D>("SimHitOccu2", "Sim Hit Occupnacy", 2 * etaMax_ + 1, -etaMax_, etaMax_ + 1, 360, 0, 360);
  hsi3Oc_ = fs_->make<TH2D>("SimHitOccu3", "Sim Hit Occupnacy", 2 * etaMax_ + 1, -etaMax_, etaMax_ + 1, 40, 0, 20);
  //Histograms for Digis
  hdigEn_ = fs_->make<TH1D>("DigiEnergy", "Digi ADC Sample", 1000, 0.0, 1000.0);
  hdigLn_ = fs_->make<TH1D>("DigiLong", "Digi Long. Profile", 40, 0.0, 20.0);
  hdigOc_ = fs_->make<TH2D>("DigiOccup", "Digi Occupnacy", 2 * etaMax_ + 1, -etaMax_, etaMax_ + 1, 360, 0, 360);
  hdi2Oc_ = fs_->make<TH2D>("DigiOccu2", "Digi Occupnacy", 2 * etaMax_ + 1, -etaMax_, etaMax_ + 1, 360, 0, 360);
  hdi3Oc_ = fs_->make<TH2D>("DigiOccu3", "Digi Occupnacy", 2 * etaMax_ + 1, -etaMax_, etaMax_ + 1, 40, 0, 20);
}

void HGCalBHValidation::analyze(const edm::Event& e, const edm::EventSetup&) {
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation:Run = " << e.id().run() << " Event = " << e.id().event();

  //SimHits
  edm::Handle<edm::PCaloHitContainer> hitsHcal;
  e.getByToken(tok_hits_, hitsHcal);
  edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation.: PCaloHitContainer"
                                      << " obtained with flag " << hitsHcal.isValid();
  if (hitsHcal.isValid()) {
    edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation: PCaloHit buffer " << hitsHcal->size();
    unsigned i(0);
    std::map<unsigned int, double> map_try;
    for (edm::PCaloHitContainer::const_iterator it = hitsHcal->begin(); it != hitsHcal->end(); ++it) {
      double energy = it->energy();
      double time = it->time();
      unsigned int id = it->id();
      int subdet, z, depth, eta, phi, lay;
      bool hbhe, bh;
      if (geomType_ == 0) {
        HcalTestNumbering::unpackHcalIndex(id, subdet, z, depth, eta, phi, lay);
        if (z == 0)
          eta = -eta;
        hbhe = ((subdet == static_cast<int>(HcalEndcap)) || (subdet == static_cast<int>(HcalBarrel)));
        bh = (subdet == static_cast<int>(HcalEndcap));
      } else {
        hbhe = bh = (DetId(id).det() == DetId::HGCalHSc);
        if (bh) {
          eta = HGCScintillatorDetId(id).ieta();
          phi = HGCScintillatorDetId(id).iphi();
          lay = HGCScintillatorDetId(id).layer();
        }
      }
      if (hbhe)
        hsi2Oc_->Fill((eta + 0.1), (phi - 0.1), energy);
      if (bh) {
        hsimE1_->Fill(energy);
        hsimTm_->Fill(time, energy);
        hsimOc_->Fill((eta + 0.1), (phi - 0.1), energy);
        hsimLn_->Fill(lay, energy);
        hsi3Oc_->Fill((eta + 0.1), lay, energy);
        double ensum(0);
        if (map_try.count(id) != 0)
          ensum = map_try[id];
        ensum += energy;
        map_try[id] = ensum;
        ++i;
        edm::LogVerbatim("HGCalValidation")
            << "HGCalBHHit[" << i << "] ID " << std::hex << " " << id << std::dec << " SubDet " << subdet << " depth "
            << depth << " Eta " << eta << " Phi " << phi << " layer " << lay << " E " << energy << " time " << time;
      }
    }
    for (std::map<unsigned int, double>::iterator itr = map_try.begin(); itr != map_try.end(); ++itr) {
      hsimE2_->Fill((*itr).second);
    }
  }

  //Digits
  unsigned int kount(0);
  if ((geomType_ == 0) && ifHCAL_) {
    edm::Handle<QIE11DigiCollection> hbhecoll;
    e.getByToken(tok_hbhe_, hbhecoll);
    edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation.: "
                                        << "HBHEQIE11DigiCollection obtained "
                                        << "with flag " << hbhecoll.isValid();
    if (hbhecoll.isValid()) {
      edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation: HBHEDigit "
                                          << "buffer " << hbhecoll->size();
      for (QIE11DigiCollection::const_iterator it = hbhecoll->begin(); it != hbhecoll->end(); ++it) {
        QIE11DataFrame df(*it);
        HcalDetId cell(df.id());
        bool hbhe =
            ((cell.subdetId() == static_cast<int>(HcalEndcap)) || (cell.subdetId() == static_cast<int>(HcalBarrel)));
        if (hbhe) {
          bool bh = (cell.subdetId() == static_cast<int>(HcalEndcap));
          int depth = cell.depth();
          double energy = df[iSample_].adc();
          analyzeDigi(cell, energy, bh, depth, kount);
        }
      }
    }
  } else {
    edm::Handle<HGCalDigiCollection> hbhecoll;
    e.getByToken(tok_hbhe_, hbhecoll);
    edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation.: "
                                        << "HGCalDigiCollection obtained with"
                                        << " flag " << hbhecoll.isValid();
    if (hbhecoll.isValid()) {
      edm::LogVerbatim("HGCalValidation") << "HGCalBHValidation: HGCalDigi "
                                          << "buffer " << hbhecoll->size();
      for (HGCalDigiCollection::const_iterator it = hbhecoll->begin(); it != hbhecoll->end(); ++it) {
        HGCalDataFrame df(*it);
        double energy = df[iSample_].data();
        if (geomType_ == 0) {
          HcalDetId cell(df.id());
          bool hbhe =
              ((cell.subdetId() == static_cast<int>(HcalEndcap)) || (cell.subdetId() == static_cast<int>(HcalBarrel)));
          if (hbhe) {
            bool bh = (cell.subdetId() == static_cast<int>(HcalEndcap));
            int depth = cell.depth();
            analyzeDigi(cell, energy, bh, depth, kount);
          }
        } else {
          bool bh = (DetId(df.id()).det() == DetId::HGCalHSc);
          if (bh) {
            HGCScintillatorDetId cell(df.id());
            int depth = cell.layer();
            analyzeDigi(cell, energy, bh, depth, kount);
          }
        }
      }
    }
  }
}

template <class T>
void HGCalBHValidation::analyzeDigi(
    const T& cell, double const& energy, bool const& bh, int const& depth, unsigned int& kount) {
  if (energy > threshold_) {
    int eta = cell.ieta();
    int phi = cell.iphi();
    hdi2Oc_->Fill((eta + 0.1), (phi - 0.1));
    if (bh) {
      hdigEn_->Fill(energy);
      hdigOc_->Fill((eta + 0.1), (phi - 0.1));
      hdigLn_->Fill(depth);
      hdi3Oc_->Fill((eta + 0.1), depth);
      ++kount;
      edm::LogVerbatim("HGCalValidation")
          << "HGCalBHDigit[" << kount << "] ID " << cell << " E " << energy << ":" << (energy > threshold_);
    }
  }
}

DEFINE_FWK_MODULE(HGCalBHValidation);
