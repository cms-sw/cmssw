// Truth dump for the upstream (beamline -> hit 0) material rule of the broken-line fits: for every reco track
// of the given collections matched to a TrackingParticle, the TP's production vertex and its tracker simhits
// (signal event only, by SimTrack id), the layers holding a cluster from that TP, the reco track's hits with
// their global positions, and the reco/TP impact parameters at the beamspot. What the fit charges upstream of
// hit 0 and what the particle crossed are then integrated offline from the material map on this geometry.
#include <cmath>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
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
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "TTree.h"

class BLUpstreamTruthDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BLUpstreamTruthDump(edm::ParameterSet const& cfg)
      : geomToken_(esConsumes()),
        topoToken_(esConsumes()),
        tpToken_(consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticles"))),
        assocToken_(consumes<reco::TrackToTrackingParticleAssociator>(cfg.getParameter<edm::InputTag>("associator"))),
        clusterTPToken_(consumes<ClusterTPAssociation>(cfg.getParameter<edm::InputTag>("clusterTPAssociation"))),
        pixelClusterToken_(
            consumes<edmNew::DetSetVector<SiPixelCluster>>(cfg.getParameter<edm::InputTag>("pixelClusters"))),
        otClusterToken_(consumes<edmNew::DetSetVector<Phase2TrackerCluster1D>>(
            cfg.getParameter<edm::InputTag>("outerTrackerClusters"))),
        beamSpotToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))) {
    usesResource(TFileService::kSharedResource);
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("tracks"))
      trackTokens_.push_back(consumes<edm::View<reco::Track>>(tag));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("simHits"))
      simHitTokens_.push_back(consumes<edm::PSimHitContainer>(tag));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("tracks", {});
    desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
    desc.add<edm::InputTag>("associator", edm::InputTag("hltTrackAssociatorByHits"));
    desc.add<edm::InputTag>("clusterTPAssociation", edm::InputTag("hltTPClusterProducer"));
    desc.add<edm::InputTag>("pixelClusters", edm::InputTag("hltSiPixelClusters"));
    desc.add<edm::InputTag>("outerTrackerClusters", edm::InputTag("hltSiPhase2Clusters"));
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
    desc.add<std::vector<edm::InputTag>>("simHits", {});
    descriptions.addWithDefaultLabel(desc);
  }

  void beginJob() override {
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("tracks", "TP-matched reco tracks with truth hits and layers");
    tree_->Branch("ev", &ev_);
    tree_->Branch("coll", &coll_);
    tree_->Branch("pt", &pt_);
    tree_->Branch("eta", &eta_);
    tree_->Branch("phi", &phi_);
    tree_->Branch("q", &q_);
    tree_->Branch("chi2", &chi2_);
    tree_->Branch("ndof", &ndof_);
    tree_->Branch("hp", &hp_);
    tree_->Branch("dxy", &dxy_);
    tree_->Branch("dz", &dz_);
    tree_->Branch("edxy", &edxy_);
    tree_->Branch("edz", &edz_);
    tree_->Branch("ept", &ept_);
    tree_->Branch("ephi", &ephi_);
    tree_->Branch("elam", &elam_);
    tree_->Branch("vx", &vx_);
    tree_->Branch("vy", &vy_);
    tree_->Branch("vz", &vz_);
    tree_->Branch("hx", &hx_);
    tree_->Branch("hy", &hy_);
    tree_->Branch("hz", &hz_);
    tree_->Branch("hsub", &hsub_);
    tree_->Branch("hlay", &hlay_);
    tree_->Branch("quality", &quality_);
    tree_->Branch("nshared", &nshared_);
    tree_->Branch("tpdg", &tpdg_);
    tree_->Branch("tpt", &tpt_);
    tree_->Branch("teta", &teta_);
    tree_->Branch("tphi", &tphi_);
    tree_->Branch("tq", &tq_);
    tree_->Branch("tvx", &tvx_);
    tree_->Branch("tvy", &tvy_);
    tree_->Branch("tvz", &tvz_);
    tree_->Branch("tdxy", &tdxy_);
    tree_->Branch("tdz", &tdz_);
    tree_->Branch("tsignal", &tsignal_);
    tree_->Branch("tnlay", &tnlay_);
    tree_->Branch("sr", &sr_);
    tree_->Branch("sz", &sz_);
    tree_->Branch("sx", &sx_);
    tree_->Branch("sy", &sy_);
    tree_->Branch("sp", &sp_);
    tree_->Branch("ssub", &ssub_);
    tree_->Branch("slay", &slay_);
    tree_->Branch("csub", &csub_);
    tree_->Branch("clay", &clay_);
    tree_->Branch("cr", &cr_);
    tree_->Branch("cz", &cz_);
  }

  void analyze(edm::Event const& ev, edm::EventSetup const& es) override {
    auto const& geom = es.getData(geomToken_);
    auto const& topo = es.getData(topoToken_);
    auto const& bs = ev.get(beamSpotToken_);
    auto tpHandle = ev.getHandle(tpToken_);
    auto const& assoc = ev.get(assocToken_);
    ev_ = ev.id().event();

    // signal-event simhits by SimTrack id (time-ordered later)
    struct SHit {
      float r, z, x, y, p, tof;
      int sub, lay;
    };
    std::unordered_map<unsigned int, std::vector<SHit>> simHits;
    for (auto const& tok : simHitTokens_) {
      auto h = ev.getHandle(tok);
      if (!h.isValid())
        continue;
      for (auto const& sh : *h) {
        DetId id(sh.detUnitId());
        auto const* det = geom.idToDet(id);
        if (det == nullptr)
          continue;
        auto g = det->surface().toGlobal(sh.entryPoint());
        simHits[sh.trackId()].push_back({float(g.perp()),
                                         float(g.z()),
                                         float(g.x()),
                                         float(g.y()),
                                         float(sh.pabs()),
                                         float(sh.timeOfFlight()),
                                         int(id.subdetId()),
                                         int(topo.layer(id))});
      }
    }

    // clusters per TP: (subdet, layer, r, z) of every cluster the truth association assigns to the TP
    struct CHit {
      int sub, lay;
      float r, z;
    };
    std::unordered_map<unsigned int, std::vector<CHit>> tpClusters;
    {
      auto const& ctp = ev.get(clusterTPToken_);
      auto const& pix = ev.get(pixelClusterToken_);
      auto const& ot = ev.get(otClusterToken_);
      // global cluster index -> DetId, for both DetSetVectors
      std::vector<unsigned int> pixDet(pix.dataSize(), 0), otDet(ot.dataSize(), 0);
      for (auto const& ds : pix)
        for (auto it = ds.begin(); it != ds.end(); ++it)
          pixDet[it - pix.data().data()] = ds.id();
      for (auto const& ds : ot)
        for (auto it = ds.begin(); it != ds.end(); ++it)
          otDet[it - ot.data().data()] = ds.id();
      for (auto const& [omni, tp] : ctp.map()) {
        if (tp.isNull())
          continue;
        unsigned int raw = 0;
        if (!omni.isValid())
          continue;
        if (omni.isPixel() && omni.index() < pixDet.size())
          raw = pixDet[omni.index()];
        else if (omni.isPhase2() && omni.index() < otDet.size())
          raw = otDet[omni.index()];
        if (raw == 0)
          continue;
        DetId id(raw);
        auto const* det = geom.idToDet(id);
        if (det == nullptr)
          continue;
        auto g = det->surface().position();
        tpClusters[tp.key()].push_back({int(id.subdetId()), int(topo.layer(id)), float(g.perp()), float(g.z())});
      }
    }

    for (unsigned int ic = 0; ic < trackTokens_.size(); ++ic) {
      auto th = ev.getHandle(trackTokens_[ic]);
      if (!th.isValid())
        continue;
      auto r2s = assoc.associateRecoToSim(th, tpHandle);
      for (unsigned int it = 0; it < th->size(); ++it) {
        edm::RefToBase<reco::Track> tref(th, it);
        auto found = r2s.find(tref);
        if (found == r2s.end() || found->val.empty())
          continue;
        auto const& tp = found->val.front().first;  // best match first
        quality_ = found->val.front().second;
        auto const& trk = *tref;
        coll_ = ic;
        pt_ = trk.pt(), eta_ = trk.eta(), phi_ = trk.phi(), q_ = trk.charge();
        chi2_ = trk.chi2(), ndof_ = trk.ndof();
        hp_ = trk.quality(reco::TrackBase::highPurity);
        math::XYZPoint bsp(bs.x0(), bs.y0(), bs.z0());
        dxy_ = trk.dxy(bsp), dz_ = trk.dz(bsp);
        edxy_ = trk.dxyError(), edz_ = trk.dzError(), ept_ = trk.ptError(), ephi_ = trk.phiError();
        elam_ = trk.lambdaError();
        vx_ = trk.vx(), vy_ = trk.vy(), vz_ = trk.vz();
        hx_.clear(), hy_.clear(), hz_.clear(), hsub_.clear(), hlay_.clear();
        nshared_ = 0;
        for (auto const& hit : trk.recHits()) {
          if (!hit->isValid())
            continue;
          auto const* bh = dynamic_cast<BaseTrackerRecHit const*>(hit);
          if (bh == nullptr)
            continue;
          auto g = bh->globalPosition();
          DetId id = hit->geographicalId();
          hx_.push_back(g.x()), hy_.push_back(g.y()), hz_.push_back(g.z());
          hsub_.push_back(int(id.subdetId())), hlay_.push_back(int(topo.layer(id)));
        }
        tpdg_ = tp->pdgId(), tpt_ = tp->pt(), teta_ = tp->eta(), tphi_ = tp->phi(), tq_ = tp->charge();
        tvx_ = tp->vx(), tvy_ = tp->vy(), tvz_ = tp->vz();
        tdxy_ = TrackingParticleIP::dxy(tp->vertex(), tp->momentum(), bsp);
        tdz_ = TrackingParticleIP::dz(tp->vertex(), tp->momentum(), bsp);
        tsignal_ = (tp->eventId().event() == 0 && tp->eventId().bunchCrossing() == 0);
        tnlay_ = tp->numberOfTrackerLayers();
        sr_.clear(), sz_.clear(), sx_.clear(), sy_.clear(), sp_.clear(), ssub_.clear(), slay_.clear();
        if (tsignal_) {
          std::vector<SHit> all;
          for (auto const& g4 : tp->g4Tracks()) {
            auto f = simHits.find(g4.trackId());
            if (f != simHits.end())
              all.insert(all.end(), f->second.begin(), f->second.end());
          }
          std::sort(all.begin(), all.end(), [](SHit const& a, SHit const& b) { return a.tof < b.tof; });
          for (auto const& s : all) {
            sr_.push_back(s.r), sz_.push_back(s.z), sx_.push_back(s.x), sy_.push_back(s.y), sp_.push_back(s.p);
            ssub_.push_back(s.sub), slay_.push_back(s.lay);
          }
        }
        csub_.clear(), clay_.clear(), cr_.clear(), cz_.clear();
        auto fc = tpClusters.find(tp.key());
        if (fc != tpClusters.end())
          for (auto const& c : fc->second) {
            csub_.push_back(c.sub), clay_.push_back(c.lay), cr_.push_back(c.r), cz_.push_back(c.z);
          }
        tree_->Fill();
      }
    }
  }

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> assocToken_;
  const edm::EDGetTokenT<ClusterTPAssociation> clusterTPToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusterToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D>> otClusterToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
  TTree* tree_ = nullptr;
  int ev_ = 0, coll_ = 0, q_ = 0, ndof_ = 0, tpdg_ = 0, tq_ = 0, tnlay_ = 0, nshared_ = 0;
  bool hp_ = false, tsignal_ = false;
  float pt_ = 0, eta_ = 0, phi_ = 0, chi2_ = 0, dxy_ = 0, dz_ = 0, edxy_ = 0, edz_ = 0, ept_ = 0, ephi_ = 0, elam_ = 0,
        vx_ = 0, vy_ = 0, vz_ = 0, quality_ = 0;
  float tpt_ = 0, teta_ = 0, tphi_ = 0, tvx_ = 0, tvy_ = 0, tvz_ = 0, tdxy_ = 0, tdz_ = 0;
  std::vector<float> hx_, hy_, hz_, sr_, sz_, sx_, sy_, sp_, cr_, cz_;
  std::vector<int> hsub_, hlay_, ssub_, slay_, csub_, clay_;
};

DEFINE_FWK_MODULE(BLUpstreamTruthDump);
