#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HcalSimHitDump : public edm::one::EDAnalyzer<> {
public:
  HcalSimHitDump(const edm::ParameterSet& ps);
  ~HcalSimHitDump() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void endJob() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void analyzeHits(std::vector<PCaloHit>&);

private:
  const std::string g4Label_, hitLab_;
  const int maxEvent_;
  const bool testNumber_;
  edm::EDGetTokenT<edm::PCaloHitContainer> toks_calo_;
  int nevt_;
};

HcalSimHitDump::HcalSimHitDump(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("ModuleLabel")),
      hitLab_(ps.getParameter<std::string>("HCCollection")),
      maxEvent_(ps.getParameter<int>("MaxEvent")),
      testNumber_(ps.getParameter<bool>("TestNumber")),
      nevt_(0) {
  // register for data access
  toks_calo_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hitLab_));

  edm::LogVerbatim("HitStudy") << "HcalSimHitDump::Module Label: " << g4Label_ << "   Hits: " << hitLab_ << " MaxEvent "
                               << maxEvent_ << " TestNumbering " << testNumber_;
}

void HcalSimHitDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::string>("HCCollection", "HcalHits");
  desc.add<int>("MaxEvent", 10);
  desc.add<bool>("TestNumber", true);
  descriptions.add("hcalSimHitDump", desc);
}

void HcalSimHitDump::analyze(const edm::Event& e, const edm::EventSetup&) {
  ++nevt_;
  edm::LogVerbatim("HitStudy") << "HcalSimHitDump::Serial # " << nevt_ << " Run # " << e.id().run() << " Event # "
                               << e.id().event();

  if (nevt_ <= maxEvent_) {
    std::vector<PCaloHit> hcHits;
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByToken(toks_calo_, hitsCalo);
    if (hitsCalo.isValid()) {
      edm::LogVerbatim("HitStudy") << "HcalValidation: get valid hist for Hcal";
      std::vector<PCaloHit> caloHits;
      caloHits.insert(caloHits.end(), hitsCalo->begin(), hitsCalo->end());
      edm::LogVerbatim("HitStudy") << "HcalValidation: Hit buffer " << caloHits.size();
      analyzeHits(caloHits);
    }
  }
}

void HcalSimHitDump::analyzeHits(std::vector<PCaloHit>& hits) {
  //Now the dump
  for (unsigned int i = 0; i < hits.size(); i++) {
    double edep = hits[i].energy();
    double time = hits[i].time();
    unsigned int id_ = hits[i].id();
    if (testNumber_) {
      int det, z, depth, eta, phi, lay;
      HcalTestNumbering::unpackHcalIndex(id_, det, z, depth, eta, phi, lay);
      std::string sub("HX");
      if (det == 1)
        sub = "HB";
      else if (det == 2)
        sub = "HE";
      else if (det == 3)
        sub = "HO";
      else if (det == 4)
        sub = "HF";
      else if (det == 5)
        sub = "HT";
      int side = (z == 0) ? (-1) : (1);
      edm::LogVerbatim("HitStudy") << "[" << i << "] (" << sub << " " << side * eta << "," << phi << "," << depth << ","
                                   << lay << ") E " << edep << " T " << time;
    } else {
      edm::LogVerbatim("HitStudy") << "[" << i << "] " << HcalDetId(id_) << " E " << edep << " T " << time;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalSimHitDump);
