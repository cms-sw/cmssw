#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <string>
#include <vector>

class EcalSimHitDump : public edm::one::EDAnalyzer<> {
public:
  EcalSimHitDump(const edm::ParameterSet& ps);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const std::string g4Label_;
  const std::vector<std::string> hitLab_;
  const std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> toksCalo_;
  const std::vector<int> types_;
  const int maxEvent_;
  int kount_;
};

EcalSimHitDump::EcalSimHitDump(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("ModuleLabel")),
      hitLab_(ps.getParameter<std::vector<std::string>>("HitCollections")),
      toksCalo_{edm::vector_transform(hitLab_,
                                      [this](const std::string& name) {
                                        return consumes<edm::PCaloHitContainer>(edm::InputTag{g4Label_, name});
                                      })},
      types_(ps.getParameter<std::vector<int>>("CollectionTypes")),
      maxEvent_(ps.getParameter<int>("MaxEvent")),
      kount_(0) {
  edm::LogVerbatim("HitStudy") << "Module Label: " << g4Label_ << "   with " << hitLab_.size()
                               << " collections and maxEvent = " << maxEvent_;
  for (unsigned int k = 0; k < hitLab_.size(); ++k)
    edm::LogVerbatim("HitStudy") << "[" << k << "] Type " << types_[k] << " Label " << hitLab_[k];
}

void EcalSimHitDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> coll = {"EcalHitsEB", "EcalHitsEE", "EcalHitsES"};
  std::vector<int> type = {0, 1, 2};
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::vector<std::string>>("HitCollections", coll);
  desc.add<std::vector<int>>("CollectionTypes", type);
  desc.add<int>("MaxEvent", 10);
  descriptions.add("ecalSimHitDump", desc);
}

void EcalSimHitDump::analyze(const edm::Event& e, const edm::EventSetup&) {
  ++kount_;
  edm::LogVerbatim("HitStudy") << "[" << kount_ << "] Run = " << e.id().run() << " Event = " << e.id().event();

  if ((kount_ <= maxEvent_) || (maxEvent_ <= 0)) {
    for (unsigned int k = 0; k < toksCalo_.size(); ++k) {
      edm::Handle<edm::PCaloHitContainer> hitsCalo;
      e.getByToken(toksCalo_[k], hitsCalo);
      if (hitsCalo.isValid())
        edm::LogVerbatim("HitStudy") << "EcalSimHitDump: Input " << hitsCalo->size() << " hits of type " << types_[k];
      unsigned int i(0);
      for (auto const& hit : *hitsCalo) {
        double edep = hit.energy();
        double time = hit.time();
        unsigned int id = hit.id();
        if (types_[k] == 0)
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << EBDetId(id) << " E" << edep << " T " << time;
        else if (types_[k] == 1)
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << EEDetId(id) << " E" << edep << " T " << time;
        else
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << ESDetId(id) << " E" << edep << " T " << time;
        ++i;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalSimHitDump);
