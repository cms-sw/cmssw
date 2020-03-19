#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

class HcalSimHitDump : public edm::EDAnalyzer {
public:
  HcalSimHitDump(const edm::ParameterSet& ps);
  ~HcalSimHitDump() override {}

protected:
  void beginJob() override {}
  void endJob() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void analyzeHits(std::vector<PCaloHit>&);

private:
  std::string g4Label_, hitLab_;
  edm::EDGetTokenT<edm::PCaloHitContainer> toks_calo_;
  int nevt_, maxEvent_;
};

HcalSimHitDump::HcalSimHitDump(const edm::ParameterSet& ps) : nevt_(0) {
  g4Label_ = ps.getUntrackedParameter<std::string>("ModuleLabel", "g4SimHits");
  hitLab_ = ps.getUntrackedParameter<std::string>("HCCollection", "HcalHits");
  maxEvent_ = ps.getUntrackedParameter<int>("MaxEvent", 10);

  // register for data access
  toks_calo_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hitLab_));

  edm::LogVerbatim("HitStudy") << "HcalSimHitDump::Module Label: " << g4Label_ << "   Hits: " << hitLab_ << " MaxEvent "
                               << maxEvent_;
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
  bool testN(false);
  for (unsigned int k = 1; k < hits.size(); ++k) {
    int det = (((hits[k].id()) >> 28) & 0xF);
    if (det != 4) {
      testN = true;
      break;
    }
  }
  edm::LogVerbatim("HitStudy") << "Hit ID uses numbering scheme " << testN << " (0 normal; 1 test)";

  //Now the dump
  for (unsigned int i = 0; i < hits.size(); i++) {
    double edep = hits[i].energy();
    double time = hits[i].time();
    unsigned int id_ = hits[i].id();
    if (testN) {
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
