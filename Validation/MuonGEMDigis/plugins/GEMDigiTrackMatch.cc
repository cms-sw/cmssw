#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>
#include "Validation/MuonGEMDigis/plugins/GEMDigiTrackMatch.h"
#include "Validation/MuonGEMHits/interface/GEMDetLabel.h"

using namespace std;
using namespace GEMDetLabel;

GEMDigiTrackMatch::GEMDigiTrackMatch(const edm::ParameterSet& ps) : GEMTrackMatch(ps) {
  std::string simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel");
  simTracksToken_ = consumes<edm::SimTrackContainer>(ps.getParameter<edm::InputTag>("simTrackCollection"));
  simVerticesToken_ = consumes<edm::SimVertexContainer>(ps.getParameter<edm::InputTag>("simVertexCollection"));

  gemDigiMatcher_.reset(new GEMDigiMatcher(ps, consumesCollector()));
}

void GEMDigiTrackMatch::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& run, edm::EventSetup const& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  geom = &*hGeom;
  setGeometry(*geom);

  ibooker.setCurrentFolder("MuonGEMDigisV/GEMDigisTask");
  LogDebug("GEMDigiTrackMatch") << "ibooker set current folder\n";

  const float PI = TMath::Pi();

  nstation = geom->regions()[0]->stations().size();
  if (detailPlot_) {
    for (unsigned int j = 0; j < nstation; j++) {
      string track_eta_name = string("track_eta") + s_suffix[j];
      string track_eta_title = string("track_eta") + ";SimTrack |#eta|;# of tracks";
      track_eta[j] = ibooker.book1D(track_eta_name.c_str(), track_eta_title.c_str(), 140, minEta_, maxEta_);

      for (unsigned int k = 0; k < 3; k++) {
        string suffix = string(s_suffix[j]) + string(c_suffix[k]);
        string track_phi_name = string("track_phi") + suffix;
        string track_phi_title = string("track_phi") + suffix + ";SimTrack #phi;# of tracks";
        track_phi[j][k] = ibooker.book1D(track_phi_name.c_str(), track_phi_title.c_str(), 200, -PI, PI);
      }

      for (unsigned int i = 0; i < 4; i++) {
        string suffix = string(s_suffix[j]) + string(l_suffix[i]);
        string dg_eta_name = string("dg_eta") + suffix;
        string dg_eta_title = dg_eta_name + "; tracks |#eta|; # of tracks";
        dg_eta[i][j] = ibooker.book1D(dg_eta_name.c_str(), dg_eta_title.c_str(), 140, minEta_, maxEta_);

        string dg_sh_eta_name = string("dg_sh_eta") + suffix;
        string dg_sh_eta_title = dg_sh_eta_name + "; tracks |#eta|; # of tracks";
        dg_sh_eta[i][j] = ibooker.book1D(dg_sh_eta_name.c_str(), dg_sh_eta_title.c_str(), 140, minEta_, maxEta_);

        string pad_eta_name = string("pad_eta") + suffix;
        string pad_eta_title = pad_eta_name + "; tracks |#eta|; # of tracks";
        pad_eta[i][j] = ibooker.book1D(pad_eta_name.c_str(), pad_eta_title.c_str(), 140, minEta_, maxEta_);

        string cluster_eta_name = string("cluster_eta") + suffix;
        string cluster_eta_title = cluster_eta_name + "; tracks |#eta|; # of tracks";
        cluster_eta[i][j] = ibooker.book1D(cluster_eta_name.c_str(), cluster_eta_title.c_str(), 140, minEta_, maxEta_);

        string copad_eta_name = string("copad_eta") + suffix;
        string copad_eta_title = copad_eta_name + "; tracks |#eta|; # of tracks";
        copad_eta[i][j] = ibooker.book1D(copad_eta_name.c_str(), copad_eta_title.c_str(), 140, minEta_, maxEta_);

        for (unsigned int k = 0; k < 3; k++) {
          suffix = string(s_suffix[j]) + string(l_suffix[i]) + string(c_suffix[k]);
          string dg_phi_name = string("dg_phi") + suffix;
          string dg_phi_title = dg_phi_name + "; tracks #phi; # of tracks";
          dg_phi[i][j][k] = ibooker.book1D((dg_phi_name).c_str(), dg_phi_title.c_str(), 200, -PI, PI);

          string dg_sh_phi_name = string("dg_sh_phi") + suffix;
          string dg_sh_phi_title = dg_sh_phi_name + "; tracks #phi; # of tracks";
          dg_sh_phi[i][j][k] = ibooker.book1D((dg_sh_phi_name).c_str(), dg_sh_phi_title.c_str(), 200, -PI, PI);

          string pad_phi_name = string("pad_phi") + suffix;
          string pad_phi_title = pad_phi_name + "; tracks #phi; # of tracks";
          pad_phi[i][j][k] = ibooker.book1D((pad_phi_name).c_str(), pad_phi_title.c_str(), 200, -PI, PI);

          string cluster_phi_name = string("cluster_phi") + suffix;
          string cluster_phi_title = cluster_phi_name + "; tracks #phi; # of tracks";
          cluster_phi[i][j][k] = ibooker.book1D((cluster_phi_name).c_str(), cluster_phi_title.c_str(), 200, -PI, PI);

          string copad_phi_name = string("copad_phi") + suffix;
          string copad_phi_title = copad_phi_name + "; tracks #phi; # of tracks";
          copad_phi[i][j][k] = ibooker.book1D((copad_phi_name).c_str(), copad_phi_title.c_str(), 200, -PI, PI);
        }
      }
    }
  }
}

GEMDigiTrackMatch::~GEMDigiTrackMatch() {}

void GEMDigiTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // setup digi matcher
  gemDigiMatcher_->init(iEvent, iSetup);

  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;

  iEvent.getByToken(simTracksToken_, sim_tracks);
  iEvent.getByToken(simVerticesToken_, sim_vertices);

  const edm::SimVertexContainer& sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer& sim_trks = *sim_tracks.product();

  if (!sim_tracks.isValid() || !sim_vertices.isValid())
    return;

  MySimTrack track_;
  for (const auto& t : sim_trks) {
    if (!isSimTrackGood(t)) {
      continue;
    }

    // match hits and digis to this SimTrack
    gemDigiMatcher_->match(t, sim_vert[t.vertIndex()]);

    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    std::fill(std::begin(track_.hitOdd), std::end(track_.hitOdd), false);
    std::fill(std::begin(track_.hitEven), std::end(track_.hitEven), false);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        track_.gem_sh[i][j] = false;
        track_.gem_dg[i][j] = false;
        track_.gem_pad[i][j] = false;
        track_.gem_cluster[i][j] = false;
      }
    }

    // ** GEM SimHits ** //
    const auto& gem_sh_ids_ch = gemDigiMatcher_->muonSimHitMatcher()->chamberIds();
    for (const auto& d : gem_sh_ids_ch) {
      const GEMDetId id(d);
      if (id.chamber() % 2 == 0)
        track_.hitEven[id.station() - 1] = true;
      else if (id.chamber() % 2 == 1)
        track_.hitOdd[id.station() - 1] = true;
      else {
        std::cout << "Error to get chamber id" << std::endl;
      }

      track_.gem_sh[id.station() - 1][(id.layer() - 1)] = true;
    }

    // ** GEM Digis, Pads and CoPads ** //
    const auto& gem_dg_ids_ch = gemDigiMatcher_->chamberIdsDigi();
    for (const auto& d : gem_dg_ids_ch) {
      const GEMDetId id(d);
      track_.gem_dg[id.station() - 1][(id.layer() - 1)] = true;
    }

    const auto& gem_pad_ids_ch = gemDigiMatcher_->chamberIdsPad();
    for (const auto& d : gem_pad_ids_ch) {
      const GEMDetId id(d);
      track_.gem_pad[id.station() - 1][(id.layer() - 1)] = true;
    }

    const auto& gem_cluster_ids_ch = gemDigiMatcher_->chamberIdsCluster();
    for (const auto& d : gem_cluster_ids_ch) {
      const GEMDetId id(d);
      track_.gem_cluster[id.station() - 1][(id.layer() - 1)] = true;
    }

    FillWithTrigger(track_eta, fabs(track_.eta));
    FillWithTrigger(track_phi, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);

    FillWithTrigger(dg_sh_eta, track_.gem_sh, fabs(track_.eta));
    FillWithTrigger(dg_eta, track_.gem_dg, fabs(track_.eta));
    FillWithTrigger(pad_eta, track_.gem_pad, fabs(track_.eta));
    FillWithTrigger(cluster_eta, track_.gem_cluster, fabs(track_.eta));
    FillWithTrigger(copad_eta, track_.gem_pad, fabs(track_.eta));

    // Separate station.

    FillWithTrigger(dg_sh_phi, track_.gem_sh, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
    FillWithTrigger(dg_phi, track_.gem_dg, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
    FillWithTrigger(pad_phi, track_.gem_pad, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
    FillWithTrigger(cluster_phi, track_.gem_cluster, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
    FillWithTrigger(copad_phi, track_.gem_pad, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
  }
}
