// Truth dump for the broken-line fits' energy-loss and material model: for every signal charged particle
// (g4SimHits SimTracks) the production momentum and vertex, and for each of its tracker PSimHits the
// momentum at entry, the global (r,z) of the entry point, the time of flight and the sensor deposit.
// Momentum differences between consecutive hits are the material's ionization loss the fits model.
#include <cmath>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "TTree.h"

class BLElossTruthDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BLElossTruthDump(edm::ParameterSet const& cfg)
      : geomToken_(esConsumes()),
        topoToken_(esConsumes()),
        simTracks_(consumes<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simTracks"))),
        simVertices_(consumes<edm::SimVertexContainer>(cfg.getParameter<edm::InputTag>("simVertices"))),
        minPt_(cfg.getParameter<double>("minPt")) {
    usesResource(TFileService::kSharedResource);
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("simHits"))
      simHits_.push_back(consumes<edm::PSimHitContainer>(tag));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
    desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
    desc.add<std::vector<edm::InputTag>>("simHits", {});
    desc.add<double>("minPt", 0.5);
    descriptions.addWithDefaultLabel(desc);
  }

  void beginJob() override {
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("tracks", "signal particles with their tracker simhits");
    tree_->Branch("pdg", &pdg_);
    tree_->Branch("q", &q_);
    tree_->Branch("p", &p_);
    tree_->Branch("pt", &pt_);
    tree_->Branch("eta", &eta_);
    tree_->Branch("vx", &vx_);
    tree_->Branch("vy", &vy_);
    tree_->Branch("vz", &vz_);
    tree_->Branch("hr", &hr_);
    tree_->Branch("hz", &hz_);
    tree_->Branch("hp", &hp_);
    tree_->Branch("htof", &htof_);
    tree_->Branch("hde", &hde_);
    tree_->Branch("hsub", &hsub_);
    tree_->Branch("hlay", &hlay_);
  }

  void analyze(edm::Event const& ev, edm::EventSetup const& es) override {
    auto const& geom = es.getData(geomToken_);
    auto const& topo = es.getData(topoToken_);
    auto const& tracks = ev.get(simTracks_);
    auto const& vertices = ev.get(simVertices_);
    // hits per track id, all collections
    struct Hit {
      float r, z, p, tof, de;
      int sub, lay;
    };
    std::unordered_map<unsigned int, std::vector<Hit>> hits;
    for (auto const& tok : simHits_) {
      auto handle = ev.getHandle(tok);
      if (!handle.isValid())
        continue;  // the Phase-2 SIM writes the branch but not every collection (TIB/TID/TOB/TEC are empty)
      for (auto const& h : *handle) {
        DetId id(h.detUnitId());
        auto const* det = geom.idToDet(id);
        if (det == nullptr)
          continue;
        auto g = det->surface().toGlobal(h.entryPoint());
        hits[h.trackId()].push_back(
            {g.perp(), g.z(), h.pabs(), h.timeOfFlight(), h.energyLoss(), int(id.subdetId()), int(topo.layer(id))});
      }
    }
    for (auto const& t : tracks) {
      if (t.charge() == 0 || t.momentum().pt() < minPt_ || t.noVertex())
        continue;
      auto const& v = vertices[t.vertIndex()];
      auto it = hits.find(t.trackId());
      if (it == hits.end())
        continue;
      auto& hv = it->second;
      std::sort(hv.begin(), hv.end(), [](Hit const& a, Hit const& b) { return a.tof < b.tof; });
      pdg_ = t.type();
      q_ = int(t.charge());
      p_ = t.momentum().P();
      pt_ = t.momentum().pt();
      eta_ = t.momentum().eta();
      vx_ = v.position().x();
      vy_ = v.position().y();
      vz_ = v.position().z();
      hr_.clear(), hz_.clear(), hp_.clear(), htof_.clear(), hde_.clear(), hsub_.clear(), hlay_.clear();
      for (auto const& h : hv) {
        hr_.push_back(h.r), hz_.push_back(h.z), hp_.push_back(h.p), htof_.push_back(h.tof), hde_.push_back(h.de);
        hsub_.push_back(h.sub), hlay_.push_back(h.lay);
      }
      tree_->Fill();
    }
  }

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracks_;
  const edm::EDGetTokenT<edm::SimVertexContainer> simVertices_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHits_;
  const double minPt_;
  TTree* tree_ = nullptr;
  int pdg_ = 0, q_ = 0;
  float p_ = 0, pt_ = 0, eta_ = 0, vx_ = 0, vy_ = 0, vz_ = 0;
  std::vector<float> hr_, hz_, hp_, htof_, hde_;
  std::vector<int> hsub_, hlay_;
};

DEFINE_FWK_MODULE(BLElossTruthDump);
