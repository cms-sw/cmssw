#include "Validation/MuonGEMRecHits/plugins/GEMRecHitTrackMatch.h"
#include "Validation/MuonGEMHits/interface/GEMDetLabel.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>

using namespace std;

GEMRecHitTrackMatch::GEMRecHitTrackMatch(const edm::ParameterSet& ps) : GEMTrackMatch(ps) {
  simTracksToken_ = consumes<edm::SimTrackContainer>(ps.getParameter<edm::InputTag>("simTrackCollection"));
  simVerticesToken_ = consumes<edm::SimVertexContainer>(ps.getParameter<edm::InputTag>("simVertexCollection"));

  gemRecHitMatcher_.reset(new GEMRecHitMatcher(ps, consumesCollector()));
}

void GEMRecHitTrackMatch::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& run,
                                         edm::EventSetup const& iSetup) {
  edm::LogInfo("GEMRecHitTrackMatch") << "GEM RecHitTrackMatch :: bookHistograms" << std::endl;

  iSetup.get<MuonGeometryRecord>().get(hGeom);
  gemGeometry_ = &*hGeom;

  edm::LogInfo("GEMRecHitTrackMatch") << "GEM RecHitTrackMatch :: geom = " << gemGeometry_ << std::endl;

  ibooker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask");
  edm::LogInfo("GEMRecHitTrackMatch") << "ibooker set current folder\n";

  const float PI = TMath::Pi();

  using namespace GEMDetLabel;

  nstation = gemGeometry_->regions()[0]->stations().size();
  for (unsigned int j = 0; j < nstation; j++) {
    string track_eta_name = string("track_eta") + s_suffix.at(j);
    string track_eta_title = string("track_eta") + ";SimTrack |#eta|;Number of tracks";
    track_eta[j] = ibooker.book1D(track_eta_name.c_str(), track_eta_title.c_str(), 140, minEta_, maxEta_);

    for (unsigned int k = 0; k < c_suffix.size(); k++) {
      string suffix = string(s_suffix[j]) + string(c_suffix[k]);
      string track_phi_name = string("track_phi") + suffix;
      string track_phi_title = string("track_phi") + suffix + ";SimTrack #phi;Number of tracks";
      track_phi[j][k] = ibooker.book1D(track_phi_name.c_str(), track_phi_title.c_str(), 200, -PI, PI);
    }

    for (unsigned int i = 0; i < l_suffix.size(); i++) {
      string suffix = string(s_suffix[j]) + string(l_suffix[i]);
      string rh_eta_name = string("rh_eta") + suffix;
      string rh_eta_title = rh_eta_name + "; tracks |#eta|; Number of tracks";
      rh_eta[i][j] = ibooker.book1D(rh_eta_name.c_str(), rh_eta_title.c_str(), 140, minEta_, maxEta_);

      string rh_sh_eta_name = string("rh_sh_eta") + suffix;
      string rh_sh_eta_title = rh_sh_eta_name + "; tracks |#eta|; Number of tracks";
      rh_sh_eta[i][j] = ibooker.book1D(rh_sh_eta_name.c_str(), rh_sh_eta_title.c_str(), 140, minEta_, maxEta_);

      for (unsigned int k = 0; k < c_suffix.size(); k++) {
        suffix = string(s_suffix[j]) + string(l_suffix[i]) + string(c_suffix[k]);
        string rh_phi_name = string("rh_phi") + suffix;
        string rh_phi_title = rh_phi_name + "; tracks #phi; Number of tracks";
        rh_phi[i][j][k] = ibooker.book1D((rh_phi_name).c_str(), rh_phi_title.c_str(), 200, -PI, PI);

        string rh_sh_phi_name = string("rh_sh_phi") + suffix;
        string rh_sh_phi_title = rh_sh_phi_name + "; tracks #phi; Number of tracks";
        rh_sh_phi[i][j][k] = ibooker.book1D((rh_sh_phi_name).c_str(), rh_sh_phi_title.c_str(), 200, -PI, PI);
      }
    }
  }
}

GEMRecHitTrackMatch::~GEMRecHitTrackMatch() {}

void GEMRecHitTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // setup rechit matcher
  gemRecHitMatcher_->init(iEvent, iSetup);

  edm::LogInfo("GEMRecHitTrackMatch") << "GEM RecHitTrackMatch :: analyze" << std::endl;

  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;

  iEvent.getByToken(simTracksToken_, sim_tracks);
  iEvent.getByToken(simVerticesToken_, sim_vertices);

  if (!sim_tracks.isValid() || !sim_vertices.isValid())
    return;

  const edm::SimVertexContainer& sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer& sim_trks = *sim_tracks.product();

  MySimTrack track_;
  for (auto& t : sim_trks) {
    if (!isSimTrackGood(t)) {
      continue;
    }

    // match rechits
    gemRecHitMatcher_->match(t, sim_vert[t.vertIndex()]);

    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    std::fill(std::begin(track_.hitOdd), std::end(track_.hitOdd), false);
    std::fill(std::begin(track_.hitEven), std::end(track_.hitEven), false);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        track_.gem_sh[i][j] = false;
        track_.gem_rh[i][j] = false;
      }
    }

    // ** GEM SimHits ** //
    for (auto d : gemRecHitMatcher_->gemDigiMatcher()->muonSimHitMatcher()->chamberIds()) {
      const GEMDetId id(d);
      if (id.chamber() % 2 == 0)
        track_.hitEven[id.station() - 1] = true;
      else if (id.chamber() % 2 == 1)
        track_.hitOdd[id.station() - 1] = true;
      else {
        edm::LogInfo("GEMRecHitTrackMatch") << "Error to get chamber id" << std::endl;
      }

      track_.gem_sh[id.station() - 1][(id.layer() - 1)] = true;
    }

    // ** GEM RecHits ** //
    for (auto d : gemRecHitMatcher_->chamberIds()) {
      const GEMDetId id(d);
      track_.gem_rh[id.station() - 1][(id.layer() - 1)] = true;
    }

    FillWithTrigger(track_eta, fabs(track_.eta));
    //FillWithTrigger(track_phi, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);

    FillWithTrigger(rh_sh_eta, track_.gem_sh, fabs(track_.eta));
    FillWithTrigger(rh_eta, track_.gem_rh, fabs(track_.eta));

    // Separate station.

    // FillWithTrigger(rh_sh_phi, track_.gem_sh, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
    // FillWithTrigger(rh_phi, track_.gem_rh, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
  }
}
