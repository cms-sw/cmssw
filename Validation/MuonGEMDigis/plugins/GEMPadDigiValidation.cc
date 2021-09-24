#include "Validation/MuonGEMDigis/plugins/GEMPadDigiValidation.h"

GEMPadDigiValidation::GEMPadDigiValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMPadDigiValidation") {
  const auto& pad_pset = pset.getParameterSet("gemPadDigi");
  const auto& pad_tag = pad_pset.getParameter<edm::InputTag>("inputTag");
  pad_token_ = consumes<GEMPadDigiCollection>(pad_tag);

  const auto& simhit_pset = pset.getParameterSet("gemSimHit");
  const auto& simhit_tag = simhit_pset.getParameter<edm::InputTag>("inputTag");
  simhit_token_ = consumes<edm::PSimHitContainer>(simhit_tag);

  const auto& digisimlink_tag = pset.getParameter<edm::InputTag>("gemDigiSimLink");
  digisimlink_token_ = consumes<edm::DetSetVector<GEMDigiSimLink>>(digisimlink_tag);

  geomToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  geomTokenBeginRun_ = esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

void GEMPadDigiValidation::bookHistograms(DQMStore::IBooker& booker,
                                          edm::Run const& Run,
                                          edm::EventSetup const& setup) {
  const GEMGeometry* gem = &setup.getData(geomTokenBeginRun_);

  // NOTE Occupancy
  booker.setCurrentFolder("GEM/Pad");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    if (detail_plot_)
      me_detail_occ_zr_[region_id] = bookZROccupancy(booker, region_id, "pad", "Pad");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2(region_id, station_id);

      if (detail_plot_) {
        me_detail_occ_det_[key2] = bookDetectorOccupancy(booker, key2, station, "pad", "Pad");
        me_detail_pad_occ_det_[key2] = bookDetectorOccupancy(booker, key2, station, "sim_matched", "Matched Pad");
      }

      const auto& superChamberVec = station->superChambers();
      if (superChamberVec.empty()) {
        edm::LogError(kLogCategory_) << "Super chambers missing for region = " << region_id
                                     << " and station = " << station_id;
        continue;
      }
      const GEMSuperChamber* super_chamber = superChamberVec.front();
      if (super_chamber == nullptr) {
        edm::LogError(kLogCategory_) << "Failed to find super chamber for region = " << region_id
                                     << " and station = " << station_id;
        continue;
      }
      for (const auto& chamber : super_chamber->chambers()) {
        Int_t layer_id = chamber->id().layer();
        ME3IdsKey key3(region_id, station_id, layer_id);

        const auto& etaPartitionVec = chamber->etaPartitions();
        if (etaPartitionVec.empty() || etaPartitionVec.front() == nullptr) {
          edm::LogError(kLogCategory_)
              << "Eta partition missing or null for region, station, super chamber, chamber = (" << region_id << ", "
              << station_id << ", " << super_chamber->id() << ", " << chamber->id() << ")";
          continue;
        }
        Int_t num_pads = etaPartitionVec.front()->npads();

        me_occ_total_pad_[key3] =
            bookHist1D(booker, key3, "total_pads_per_event", "Number of pad digis per event", 51, -0.5, 50);

        me_pad_occ_eta_[key3] = bookHist1D(booker,
                                           key3,
                                           "sim_matched_occ_eta",
                                           "Matched Pad Eta Occupancy",
                                           16,
                                           eta_range_[station_id * 2 + 0],
                                           eta_range_[station_id * 2 + 1],
                                           "#eta");

        me_pad_occ_phi_[key3] =
            bookHist1D(booker, key3, "sim_matched_occ_phi", "Matched Pad Phi Occupancy", 36, -5, 355, "#phi [degrees]");

        if (detail_plot_) {
          me_detail_occ_xy_[key3] = bookXYOccupancy(booker, key3, "pad", "Pad");

          me_detail_occ_phi_pad_[key3] = bookHist2D(booker,
                                                    key3,
                                                    "occ_phi_pad",
                                                    "Pad Occupancy",
                                                    280,
                                                    -M_PI,
                                                    M_PI,
                                                    num_pads / 2,
                                                    0,
                                                    num_pads,
                                                    "#phi [rad]",
                                                    "Pad number");

          me_detail_occ_pad_[key3] =
              bookHist1D(booker, key3, "occ_pad", "Pad Occupancy", num_pads, 0.5, num_pads + 0.5, "Pad number");
        }
      }  // layer loop
    }    // station loop
  }      // region loop

  // NOTE Bunch Crossing
  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const auto& superChamberVec = station->superChambers();
        if (superChamberVec.empty()) {
          edm::LogError(kLogCategory_) << "Super chambers missing for region = " << region_id
                                       << " and station = " << station_id;
          continue;
        }
        const GEMSuperChamber* super_chamber = superChamberVec.front();
        if (super_chamber == nullptr) {
          edm::LogError(kLogCategory_) << "Failed to find super chamber for region = " << region_id
                                       << " and station = " << station_id;
          continue;
        }
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          me_detail_bx_[key3] = bookHist1D(booker, key3, "bx", "Pad Bunch Crossing", 5, -2.5, 2.5, "Bunch crossing");
        }  // chamber loop
      }    // station loop
    }      // region loop
  }        // detail plot
}

GEMPadDigiValidation::~GEMPadDigiValidation() {}

void GEMPadDigiValidation::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  const GEMGeometry* gem = &setup.getData(geomToken_);

  edm::Handle<GEMPadDigiCollection> collection;
  event.getByToken(pad_token_, collection);
  if (not collection.isValid()) {
    edm::LogError(kLogCategory_) << "Cannot get pads by label GEMPadToken.";
    return;
  }

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

  std::map<ME3IdsKey, Int_t> total_pad;
  for (const auto& pad_pair : *collection) {
    GEMDetId gemid = pad_pair.first;
    const auto& range = pad_pair.second;

    if (gem->idToDet(gemid) == nullptr) {
      edm::LogError(kLogCategory_) << "Getting DetId failed. Discard this gem pad hit. "
                                   << "Maybe it comes from unmatched geometry." << std::endl;
      continue;
    }

    const GEMEtaPartition* roll = gem->etaPartition(gemid);
    const BoundPlane& surface = roll->surface();

    Int_t region_id = gemid.region();
    Int_t station_id = gemid.station();
    Int_t layer_id = gemid.layer();
    Int_t chamber_id = gemid.chamber();
    Int_t ieta = gemid.ieta();
    Int_t num_layers = gemid.nlayers();

    ME2IdsKey key2(region_id, station_id);
    ME3IdsKey key3(region_id, station_id, layer_id);

    for (auto digi = range.first; digi != range.second; ++digi) {
      // ignore 16-partition GE2/1 pads
      if (gemid.isGE21() and digi->nPartitions() == GEMPadDigi::GE21SplitStrip)
        continue;

      total_pad[key3]++;

      Int_t pad = digi->pad();
      Int_t bx = digi->bx();

      const LocalPoint& local_pos = roll->centreOfPad(pad);
      const GlobalPoint& global_pos = surface.toGlobal(local_pos);

      Float_t g_r = global_pos.perp();
      Float_t g_phi = global_pos.phi();
      Float_t g_x = global_pos.x();
      Float_t g_y = global_pos.y();
      Float_t g_abs_z = std::fabs(global_pos.z());

      Int_t bin_x = getDetOccBinX(num_layers, chamber_id, layer_id);

      if (detail_plot_) {
        me_detail_occ_zr_[region_id]->Fill(g_abs_z, g_r);
        me_detail_occ_xy_[key3]->Fill(g_x, g_y);
        me_detail_occ_phi_pad_[key3]->Fill(g_phi, pad);
        me_detail_occ_pad_[key3]->Fill(pad);
        me_detail_occ_det_[key2]->Fill(bin_x, ieta);
        me_detail_bx_[key3]->Fill(bx);
      }  // if detail_plot
    }    // digi loop
  }      // range loop

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();
    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      const auto& superChamberVec = station->superChambers();
      if (superChamberVec.empty()) {
        edm::LogError(kLogCategory_) << "Super chambers missing for region = " << region_id
                                     << " and station = " << station_id;
        continue;
      }
      const GEMSuperChamber* super_chamber = superChamberVec.front();
      if (super_chamber == nullptr) {
        edm::LogError(kLogCategory_) << "Failed to find super chamber for region = " << region_id
                                     << " and station = " << station_id;
        continue;
      }
      for (const auto& chamber : super_chamber->chambers()) {
        Int_t layer_id = chamber->id().layer();
        ME3IdsKey key3{region_id, station_id, layer_id};
        me_occ_total_pad_[key3]->Fill(total_pad[key3]);
      }
    }
  }

  // NOTE
  for (const auto& simhit : *simhit_container.product()) {
    if (not isMuonSimHit(simhit))
      continue;
    if (gem->idToDet(simhit.detUnitId()) == nullptr) {
      edm::LogError(kLogCategory_) << "SimHit did not match with GEMGeometry." << std::endl;
      continue;
    }

    GEMDetId simhit_gemid(simhit.detUnitId());

    Int_t region_id = simhit_gemid.region();
    Int_t station_id = simhit_gemid.station();
    Int_t layer_id = simhit_gemid.layer();
    Int_t chamber_id = simhit_gemid.chamber();
    Int_t ieta = simhit_gemid.ieta();
    Int_t num_layers = simhit_gemid.nlayers();

    ME2IdsKey key2{region_id, station_id};
    ME3IdsKey key3{region_id, station_id, layer_id};

    const GEMEtaPartition* roll = gem->etaPartition(simhit_gemid);

    const auto& simhit_local_pos = simhit.localPosition();
    const auto& simhit_global_pos = roll->surface().toGlobal(simhit_local_pos);

    Float_t simhit_g_eta = std::abs(simhit_global_pos.eta());
    Float_t simhit_g_phi = toDegree(simhit_global_pos.phi());

    auto simhit_trackId = simhit.trackId();

    Int_t bin_x = getDetOccBinX(num_layers, chamber_id, layer_id);

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
    Int_t simhit_pad = roll->padOfStrip(simhit_strip);
    auto range = collection->get(simhit_gemid);
    for (auto pad = range.first; pad != range.second; ++pad) {
      if (pad->pad() == simhit_pad) {
        me_pad_occ_eta_[key3]->Fill(simhit_g_eta);
        me_pad_occ_phi_[key3]->Fill(simhit_g_phi);
        if (detail_plot_) {
          me_detail_pad_occ_det_[key2]->Fill(bin_x, ieta);
        }
        break;
      }
    }
  }  // simhit_container loop
}
