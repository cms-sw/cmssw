#include "Validation/MuonGEMRecHits/plugins/GEMRecHitValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

GEMRecHitValidation::GEMRecHitValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMRecHitsValidation") {
  const auto& rechit_pset = pset.getParameterSet("gemRecHit");
  const auto& rechit_tag = rechit_pset.getParameter<edm::InputTag>("inputTag");
  rechit_token_ = consumes<GEMRecHitCollection>(rechit_tag);

  const auto& simhit_pset = pset.getParameterSet("gemSimHit");
  const auto& simhit_tag = simhit_pset.getParameter<edm::InputTag>("inputTag");

  const auto& digisimlink_tag = pset.getParameter<edm::InputTag>("gemDigiSimLink");
  digisimlink_token_ = consumes<edm::DetSetVector<GEMDigiSimLink>>(digisimlink_tag);

  simhit_token_ = consumes<edm::PSimHitContainer>(simhit_tag);
  geomToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  geomTokenBeginRun_ = esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

GEMRecHitValidation::~GEMRecHitValidation() {}

void GEMRecHitValidation::bookHistograms(DQMStore::IBooker& booker, edm::Run const& run, edm::EventSetup const& setup) {
  const GEMGeometry* gem = &setup.getData(geomTokenBeginRun_);

  // NOTE Cluster Size
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/ClusterSize");

  TString cls_title = "Cluster Size Distribution";
  TString cls_x_title = "Cluster size";

  me_cls_ = booker.book1D("cls", cls_title + ";" + cls_x_title + ";" + "Entries", 10, 0.5, 10.5);

  for (const auto& station : gem->regions()[0]->stations()) {
    Int_t station_id = station->station();
    for (const auto& roll : station->superChambers()[0]->chambers()[0]->etaPartitions()) {
      Int_t ieta = roll->id().ieta();
      ME2IdsKey key{station_id, ieta};
      me_cls_roll_[key] = booker.book1D(Form("cls_GE%d1_iEta%d", station_id, ieta),
                                        Form("Cluster Size Distribution : GE%d1 iEta %d", station_id, ieta),
                                        10,
                                        0.5,
                                        10.5);
    }
  }

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const auto& superChamberVec = station->superChambers();
        if (superChamberVec.empty() || superChamberVec[0] == nullptr) {
          edm::LogError(kLogCategory_) << "Super chambers missing or null for region = " << region_id
                                       << " and station = " << station_id;
        } else {
          for (const auto& chamber : superChamberVec[0]->chambers()) {
            Int_t layer_id = chamber->id().layer();

            for (const auto& roll : chamber->etaPartitions()) {
              Int_t ieta = roll->id().ieta();
              ME4IdsKey key4{region_id, station_id, layer_id, ieta};

              me_detail_cls_[key4] = bookHist1D(booker, key4, "cls", "Cluster Size Distribution", 11, -0.5, 10.5);
            }  // roll loop
          }    // chamber loop
        }      // end else
      }        // station loop
    }          // region loop
  }            // detail plot

  // NOTE Residual
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/Residual");

  for (const auto& station : gem->regions()[0]->stations()) {
    Int_t station_id = station->station();
    for (const auto& roll : station->superChambers()[0]->chambers()[0]->etaPartitions()) {
      Int_t ieta = roll->id().ieta();
      ME2IdsKey key{station_id, ieta};

      me_residual_x_[key] = booker.book1D(Form("residual_x_GE%d1_R%d", station_id, ieta),
                                          Form("Residual in X : GE%d1 iEta %d; Residual in X [cm]", station_id, ieta),
                                          60,
                                          -2,
                                          2);

      me_residual_y_[key] = booker.book1D(Form("residual_y_GE%d1_iEta%d", station_id, ieta),
                                          Form("Residual in Y : GE%d1 iEta %d; Residual in Y [cm]", station_id, ieta),
                                          60,
                                          -15,
                                          15);

      me_residual_rphi_[key] = booker.book1D(
          Form("residual_rphi_GE%d1_iEta%d", station_id, ieta),
          Form("Residual in R #times #phi : GE%d1 iEta %d; Residual in r #times #phi [cm]", station_id, ieta),
          60,
          -15,
          15);
    }
  }

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const auto& superChamberVec = station->superChambers();
        if (!superChamberVec.empty() && superChamberVec[0] != nullptr) {
          for (const auto& chamber : superChamberVec[0]->chambers()) {
            Int_t layer_id = chamber->id().layer();

            for (const auto& roll : chamber->etaPartitions()) {
              Int_t ieta = roll->id().ieta();
              ME4IdsKey key4{region_id, station_id, layer_id, ieta};

              me_detail_residual_x_[key4] =
                  bookHist1D(booker, key4, "residual_x", "Residual in x", 60, -2, 2, "Residual in x [cm]");

              me_detail_residual_y_[key4] =
                  bookHist1D(booker, key4, "residual_y", "Residual in y", 60, -15, 15, "Residual in y [cm]");

              me_detail_residual_rphi_[key4] = bookHist1D(booker,
                                                          key4,
                                                          "residual_rphi",
                                                          "Residual in r #times #phi",
                                                          60,
                                                          -15,
                                                          15,
                                                          "Residual in r #times #phi [cm]");
            }  // roll loop
          }    // chamber loop
        }      // end if
      }        // station loop
    }          // region loop
  }            // detail plot

  // NOTE Pull
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/Pull");

  if (detail_plot_) {
    for (const auto& station : gem->regions()[0]->stations()) {
      Int_t station_id = station->station();
      for (const auto& roll : station->superChambers()[0]->chambers()[0]->etaPartitions()) {
        Int_t ieta = roll->id().ieta();
        ME2IdsKey key{station_id, ieta};

        me_detail_pull_x_[key] = booker.book1D(Form("pull_x_GE%d1_iEta%d", station_id, ieta),
                                               Form("Pull in X : GE%d1 iEta %d", station_id, ieta),
                                               60,
                                               -3,
                                               3);

        me_detail_pull_y_[key] = booker.book1D(Form("pull_y_GE%d1_iEta%d", station_id, ieta),
                                               Form("Pull in Y : GE%d1 iEta %d", station_id, ieta),
                                               60,
                                               -3,
                                               3);
      }
    }

    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const auto& superChamberVec = station->superChambers();
        if (!superChamberVec.empty() && superChamberVec[0] != nullptr) {
          for (const auto& chamber : superChamberVec[0]->chambers()) {
            Int_t layer_id = chamber->id().layer();

            for (const auto& roll : chamber->etaPartitions()) {
              Int_t ieta = roll->id().ieta();
              ME4IdsKey key4{region_id, station_id, layer_id, ieta};

              me_detail_pull_x_la_[key4] = bookHist1D(booker, key4, "pull_x", "Pull in x", 60, -3, 3);

              me_detail_pull_y_la_[key4] = bookHist1D(booker, key4, "pull_y", "Pull in y", 60, -3, 3);
            }  // roll loop
          }    // chamber loop
        }      // end if
      }        // station loop
    }          // region loop
  }            // detail plot

  // NOTE Occupancy
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/Occupancy");
  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    if (detail_plot_)
      me_detail_occ_zr_[region_id] = bookZROccupancy(booker, region_id, "rechit", "RecHit");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2{region_id, station_id};

      if (detail_plot_)
        me_detail_rechit_occ_det_[key2] =
            bookDetectorOccupancy(booker, key2, station, "matched_rechit", "Matched RecHit");

      const auto& superChamberVec = station->superChambers();
      if (!superChamberVec.empty() && superChamberVec[0] != nullptr) {
        for (const auto& chamber : superChamberVec[0]->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3{region_id, station_id, layer_id};

          Int_t num_eta_partitions = chamber->nEtaPartitions();

          me_total_rechit_[key3] =
              bookHist1D(booker, key3, "total_rechit", "Number of rec hits per event", 25, -0.5, 24.5);

          me_occ_pid_[key3] = bookPIDHist(booker, key3, "rechit_occ_pid", "Number of entreis for each particle");

          me_occ_ieta_[key3] = bookHist1D(booker,
                                          key3,
                                          "rechit_occ_ieta",
                                          "Rechit Occupancy per eta partition",
                                          num_eta_partitions,
                                          0.5,
                                          num_eta_partitions + 0.5);

          me_occ_phi_[key3] = bookHist1D(booker, key3, "rechit_occ_phi", "Rechit Phi Occupancy", 108, -5, 355);

          me_rechit_occ_eta_[key3] = bookHist1D(booker,
                                                key3,
                                                "matched_rechit_occ_eta",
                                                "Matched RecHit Eta Occupancy",
                                                16,
                                                eta_range_[station_id * 2 + 0],
                                                eta_range_[station_id * 2 + 1],
                                                "|#eta|");

          me_rechit_occ_phi_[key3] =
              bookHist1D(booker, key3, "matched_rechit_occ_phi", "Matched RecHit Phi Occupancy", 36, -5, 355, "#phi");

          if (detail_plot_) {
            me_detail_occ_xy_[key3] = bookXYOccupancy(booker, key3, "rechit", "RecHit");

            me_detail_occ_polar_[key3] = bookPolarOccupancy(booker, key3, "rechit", "RecHit");
          }
        }  // chamber loop
      }    // end if
    }      // station loop
  }        // region_loop
}

Bool_t GEMRecHitValidation::matchRecHitAgainstSimHit(GEMRecHitCollection::const_iterator rechit, Int_t simhit_strip) {
  Int_t cls = rechit->clusterSize();
  Int_t rechit_first_strip = rechit->firstClusterStrip();

  if (cls == 1) {
    return simhit_strip == rechit_first_strip;
  } else {
    Int_t rechit_last_strip = rechit_first_strip + cls - 1;
    return (simhit_strip >= rechit_first_strip) and (simhit_strip <= rechit_last_strip);
  }
}

void GEMRecHitValidation::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  const GEMGeometry* gem = &setup.getData(geomToken_);

  edm::Handle<edm::DetSetVector<GEMDigiSimLink>> digiSimLink;
  event.getByToken(digisimlink_token_, digiSimLink);
  if (not digiSimLink.isValid()) {
    edm::LogError(kLogCategory_) << "Failed to get GEMDigiSimLink." << std::endl;
    return;
  }

  edm::Handle<edm::PSimHitContainer> simhit_container;
  event.getByToken(simhit_token_, simhit_container);
  if (not simhit_container.isValid()) {
    edm::LogError(kLogCategory_) << "Failed to get PSimHitContainer." << std::endl;
    return;
  }

  edm::Handle<GEMRecHitCollection> rechit_collection;
  event.getByToken(rechit_token_, rechit_collection);
  if (not rechit_collection.isValid()) {
    edm::LogError(kLogCategory_) << "Failed to get GEMRecHitCollection" << std::endl;
    return;
  }

  std::map<ME3IdsKey, Int_t> total_rechit;
  for (const auto& rechit : *rechit_collection) {
    GEMDetId gem_id{rechit.gemId()};
    Int_t region_id = gem_id.region();
    Int_t station_id = gem_id.station();
    Int_t layer_id = gem_id.layer();
    Int_t ieta = gem_id.ieta();
    ME4IdsKey key4{region_id, station_id, layer_id, ieta};

    ME2IdsKey key{station_id, ieta};
    ME2IdsKey key2{region_id, station_id};
    ME3IdsKey key3{region_id, station_id, layer_id};

    const BoundPlane& surface = gem->idToDet(gem_id)->surface();
    GlobalPoint&& rechit_global_pos = surface.toGlobal(rechit.localPosition());

    Float_t rechit_g_x = rechit_global_pos.x();
    Float_t rechit_g_y = rechit_global_pos.y();
    Float_t rechit_g_abs_z = std::fabs(rechit_global_pos.z());
    Float_t rechit_g_r = rechit_global_pos.perp();
    Float_t rechit_g_phi = toDegree(rechit_global_pos.phi());

    Int_t first_strip = rechit.firstClusterStrip();
    Int_t cls = rechit.clusterSize();
    cls = cls > 10 ? 10 : cls;

    me_cls_roll_[key]->Fill(cls);
    me_occ_ieta_[key3]->Fill(ieta);
    me_occ_phi_[key3]->Fill(rechit_g_phi);
    total_rechit[key3]++;

    me_cls_->Fill(cls);

    if (detail_plot_) {
      me_detail_cls_[key4]->Fill(cls);
      me_detail_occ_zr_[region_id]->Fill(rechit_g_abs_z, rechit_g_r);
      me_detail_occ_xy_[key3]->Fill(rechit_g_x, rechit_g_y);
      me_detail_occ_polar_[key3]->Fill(rechit_g_phi, rechit_g_r);
    }  // detail plot

    auto links = digiSimLink->find(gem_id);

    if (links == digiSimLink->end())
      continue;
    std::map<Int_t, Int_t> pid_count;

    for (Int_t strip = first_strip; strip < first_strip + cls; strip++) {
      for (const auto& link : *links) {
        Int_t link_strip = link.getStrip();
        if (link_strip == strip) {
          Int_t pid = link.getParticleType();
          pid_count[pid]++;
          break;
        }
      }
    }
    Int_t max_pid = 0;
    Int_t max_count = 0;
    for (auto& [pid, count] : pid_count) {
      if (max_count < count) {
        max_pid = pid;
        max_count = count;
      }
    }
    Int_t pid_idx = getPidIdx(max_pid);
    me_occ_pid_[key3]->Fill(pid_idx);
  }

  for (auto [key, num_total_rechit] : total_rechit) {
    me_total_rechit_[key]->Fill(num_total_rechit);
  }

  // NOTE
  for (const auto& simhit : *simhit_container.product()) {
    if (gem->idToDet(simhit.detUnitId()) == nullptr) {
      edm::LogError(kLogCategory_) << "MuonGEMHit didn't matched with GEMGeometry." << std::endl;
      continue;
    }

    GEMDetId simhit_gemid{simhit.detUnitId()};
    const BoundPlane& surface = gem->idToDet(simhit_gemid)->surface();

    Int_t region_id = simhit_gemid.region();
    Int_t station_id = simhit_gemid.station();
    Int_t layer_id = simhit_gemid.layer();
    Int_t chamber_id = simhit_gemid.chamber();
    Int_t ieta = simhit_gemid.ieta();
    Int_t num_layers = simhit_gemid.nlayers();

    ME2IdsKey key{station_id, ieta};
    ME2IdsKey key2{region_id, station_id};
    ME3IdsKey key3{region_id, station_id, layer_id};
    ME4IdsKey key4{region_id, station_id, layer_id, ieta};

    const LocalPoint& simhit_local_pos = simhit.localPosition();
    const GlobalPoint& simhit_global_pos = surface.toGlobal(simhit_local_pos);

    Float_t simhit_g_abs_eta = std::fabs(simhit_global_pos.eta());
    Float_t simhit_g_phi = toDegree(simhit_global_pos.phi());

    Int_t det_occ_bin_x = getDetOccBinX(num_layers, chamber_id, layer_id);

    auto simhit_trackId = simhit.trackId();

    auto links = digiSimLink->find(simhit_gemid);
    if (links == digiSimLink->end())
      continue;

    Int_t simhit_strip = -1;
    for (const auto& link : *links) {
      if (simhit_trackId == link.getTrackId()) {
        simhit_strip = link.getStrip();
        break;
      }
    }

    GEMRecHitCollection::range range = rechit_collection->get(simhit_gemid);
    for (auto rechit = range.first; rechit != range.second; ++rechit) {
      if (gem->idToDet(rechit->gemId()) == nullptr) {
        edm::LogError(kLogCategory_) << "GEMRecHit didn't matched with GEMGeometry." << std::endl;
        continue;
      }

      if (not isMuonSimHit(simhit))
        continue;

      if (matchRecHitAgainstSimHit(rechit, simhit_strip)) {
        const LocalPoint& rechit_local_pos = rechit->localPosition();

        Float_t resolution_x = std::sqrt(rechit->localPositionError().xx());
        Float_t resolution_y = std::sqrt(rechit->localPositionError().yy());

        Float_t residual_x = rechit_local_pos.x() - simhit_local_pos.x();
        Float_t residual_y = rechit_local_pos.y() - simhit_local_pos.y();
        Float_t residual_r = sqrt(pow(residual_x, 2) + pow(residual_y, 2));
        Float_t residual_phi = rechit_local_pos.phi() - simhit_local_pos.phi();
        Float_t residual_rphi = residual_r * residual_phi;

        Float_t pull_x = residual_x / resolution_x;
        Float_t pull_y = residual_y / resolution_y;

        me_residual_x_[key]->Fill(residual_x);
        me_residual_y_[key]->Fill(residual_y);
        me_residual_rphi_[key]->Fill(residual_rphi);

        me_rechit_occ_eta_[key3]->Fill(simhit_g_abs_eta);
        me_rechit_occ_phi_[key3]->Fill(simhit_g_phi);

        if (detail_plot_) {
          me_detail_rechit_occ_det_[key2]->Fill(det_occ_bin_x, ieta);

          me_detail_residual_x_[key4]->Fill(residual_x);
          me_detail_residual_y_[key4]->Fill(residual_y);
          me_detail_residual_rphi_[key4]->Fill(residual_rphi);

          me_detail_pull_x_[key]->Fill(pull_x);
          me_detail_pull_y_[key]->Fill(pull_y);
          me_detail_pull_x_la_[key4]->Fill(pull_x);
          me_detail_pull_y_la_[key4]->Fill(pull_y);
        }  // detail_plot

        // If we find GEMRecHit that matches PSimHit, then exit
        // GEMRecHitCollection loop.
        break;

      }  // if rechit matches against simhit
    }    // rechit loop
  }      // simhit loop
}
