#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class HcalTestSimHitID : public edm::EDAnalyzer {
public:
  HcalTestSimHitID(const edm::ParameterSet& ps);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void endJob() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  const std::string g4Label_, hitLab_;
  const bool testN_, dumpHits_;
  const int maxEvent_;
  int nevt_;
  edm::EDGetTokenT<edm::PCaloHitContainer> toks_calo_;
};

HcalTestSimHitID::HcalTestSimHitID(const edm::ParameterSet& ps)
    : g4Label_(ps.getUntrackedParameter<std::string>("moduleLabel", "g4SimHits")),
      hitLab_(ps.getUntrackedParameter<std::string>("hcCollection", "HcalHits")),
      testN_(ps.getUntrackedParameter<bool>("testNumbering", false)),
      dumpHits_(ps.getUntrackedParameter<bool>("dumpHits", false)),
      maxEvent_(ps.getUntrackedParameter<int>("maxEvent", 100)),
      nevt_(0) {
  // register for data access
  toks_calo_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hitLab_));

  std::cout << "HcalTestSimHitID::Module Label: " << g4Label_ << "   Hits: " << hitLab_ << " MaxEvent: " << maxEvent_
            << " Numbering scheme: " << testN_ << " (0 normal; 1 test)\n";
}

void HcalTestSimHitID::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("moduleLabel", "g4SimHits");
  desc.addUntracked<std::string>("hcCollection", "HcalHits");
  desc.addUntracked<bool>("testNumbering", false);
  desc.addUntracked<bool>("dumpHits", false);
  desc.addUntracked<int>("maxEvent", 100);
  descriptions.add("hcalGeometryDetIdTester", desc);
}

void HcalTestSimHitID::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  ++nevt_;
  std::cout << "HcalTestSimHitID::Serial # " << nevt_ << " Run # " << e.id().run() << " Event # " << e.id().event()
            << std::endl;
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iS.get<HcalRecNumberingRecord>().get(pHRNDC);
  const HcalDDDRecConstants* hcr = static_cast<const HcalDDDRecConstants*>(&(*pHRNDC));
  edm::ESHandle<HcalTopology> htopo;
  iS.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  if (nevt_ <= maxEvent_) {
    std::vector<PCaloHit> hcHits;
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByToken(toks_calo_, hitsCalo);
    if (hitsCalo.isValid()) {
      std::vector<PCaloHit> hits;
      hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
      std::cout << "HcalValidation: Hit buffer " << hits.size() << std::endl;

      //Now the testing
      unsigned int good(0);
      for (unsigned int i = 0; i < hits.size(); i++) {
        unsigned int id = hits[i].id();
        HcalDetId hid;
        if (testN_) {
          hid = HcalDetId(HcalHitRelabeller::relabel(id, hcr));
        } else {
          hid = HcalDetId(id);
        }
        if (theHBHETopology->validHcal(hid)) {
          ++good;
          if (dumpHits_)
            std::cout << "Hit[" << i << "] " << hid << " \n";
        } else {
          std::cout << "Hit[" << i << "] " << hid << " ***** ERROR *****\n";
        }
      }
      std::cout << "HcalTestSimHitID:: " << good << " among " << hits.size() << " hits\n";
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalTestSimHitID);
