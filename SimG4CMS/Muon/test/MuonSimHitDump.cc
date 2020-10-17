#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <string>
#include <vector>

class MuonSimHitDump : public edm::one::EDAnalyzer<> {
public:
  MuonSimHitDump(const edm::ParameterSet& ps);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const std::string g4Label_;
  const std::vector<std::string> hitLab_;
  const std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> toksMuon_;
  const std::vector<int> types_;
  const int maxEvent_;
  int kount_;
};

MuonSimHitDump::MuonSimHitDump(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("ModuleLabel")),
      hitLab_(ps.getParameter<std::vector<std::string>>("HitCollections")),
      toksMuon_{edm::vector_transform(hitLab_,
                                      [this](const std::string& name) {
                                        return consumes<edm::PSimHitContainer>(edm::InputTag{g4Label_, name});
                                      })},
      types_(ps.getParameter<std::vector<int>>("CollectionTypes")),
      maxEvent_(ps.getParameter<int>("MaxEvent")),
      kount_(0) {
  edm::LogVerbatim("HitStudy") << "Module Label: " << g4Label_ << "   with " << hitLab_.size()
                               << " collections and maxEvent = " << maxEvent_;
  for (unsigned int k = 0; k < hitLab_.size(); ++k)
    edm::LogVerbatim("HitStudy") << "[" << k << "] Type " << types_[k] << " Label " << hitLab_[k];
}

void MuonSimHitDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> coll = {"MuonDTHits", "MuonCSCHits", "MuonRPCHits", "MuonGEMHits"};
  std::vector<int> type = {0, 1, 2, 3};
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::vector<std::string>>("HitCollections", coll);
  desc.add<std::vector<int>>("CollectionTypes", type);
  desc.add<int>("MaxEvent", 10);
  descriptions.add("muonSimHitDump", desc);
}

void MuonSimHitDump::analyze(const edm::Event& e, const edm::EventSetup&) {
  ++kount_;
  edm::LogVerbatim("HitStudy") << "[" << kount_ << "] Run = " << e.id().run() << " Event = " << e.id().event();
  std::vector<std::string> dets = {"DT", "CSC", "RPC", "GEM", "ME0"};
  if ((kount_ <= maxEvent_) || (maxEvent_ <= 0)) {
    for (unsigned int k = 0; k < toksMuon_.size(); ++k) {
      edm::Handle<edm::PSimHitContainer> hitsMuon;
      e.getByToken(toksMuon_[k], hitsMuon);
      if (hitsMuon.isValid())
        edm::LogVerbatim("HitStudy") << "MuonSimHitDump: Input " << hitsMuon->size() << " hits of type " << types_[k]
                                     << " (" << dets[k] << ")";
      unsigned int i(0);
      for (auto const& hit : *hitsMuon) {
        auto entry = hit.entryPoint();
        auto exit = hit.exitPoint();
        auto p = hit.pabs();
        auto tof = hit.tof();
        auto edep = hit.energyLoss();
        unsigned int track = hit.trackId();
        unsigned int id = hit.detUnitId();
        if (types_[k] == 0)
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << DTWireId(id) << " Trk " << track << " p " << p << " dE "
                                       << edep << " T " << tof << " Enetry " << entry << " Exit " << exit;
        else if (types_[k] == 1)
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << CSCDetId(id) << " Trk " << track << " p " << p << " dE "
                                       << edep << " T " << tof << " Enetry " << entry << " Exit " << exit;
        else if (types_[k] == 2)
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << RPCDetId(id) << " Trk " << track << " p " << p << " dE "
                                       << edep << " T " << tof << " Enetry " << entry << " Exit " << exit;
        else if (types_[k] == 3)
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << GEMDetId(id) << " Trk " << track << " p " << p << " dE "
                                       << edep << " T " << tof << " Enetry " << entry << " Exit " << exit;
        else
          edm::LogVerbatim("HitStudy") << "[" << i << "] " << ME0DetId(id) << " Trk " << track << " p " << p << " dE "
                                       << edep << " T " << tof << " Enetry " << entry << " Exit " << exit;
        ++i;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonSimHitDump);
