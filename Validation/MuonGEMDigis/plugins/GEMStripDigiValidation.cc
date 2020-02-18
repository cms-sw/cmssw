#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonGEMDigis/plugins/GEMStripDigiValidation.h"


GEMStripDigiValidation::GEMStripDigiValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMStripDigiValidation") {

  const auto& strip_pset = pset.getParameterSet("gemStripDigi");
  const auto& strip_tag = strip_pset.getParameter<edm::InputTag>("inputTag");
  strip_token_ = consumes<GEMDigiCollection>(strip_tag);

  const auto& simhit_pset = pset.getParameterSet("gemSimHit");
  const auto& simhit_tag = simhit_pset.getParameter<edm::InputTag>("inputTag");
  simhit_token_ = consumes<edm::PSimHitContainer>(simhit_tag);
}


void GEMStripDigiValidation::bookHistograms(DQMStore::IBooker& booker,
                                            edm::Run const& run,
                                            edm::EventSetup const& setup) {
  const GEMGeometry* gem = initGeometry(setup);

  // NOTE Occupancy
  booker.setCurrentFolder("MuonGEMDigisV/GEMDigisTask/Strip/Occupancy");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    // occupancy plots for eta efficiency
    me_simhit_occ_eta_[region_id] = bookHist1D(
        booker, region_id, "muon_simhit_occ_eta", "Muon SimHit Eta Occupancy",
        50, eta_range_[0], eta_range_[1], "|#eta|");

    me_strip_occ_eta_[region_id] = bookHist1D(
        booker, region_id, "matched_strip_occ_eta", "Strip Digi Eta Occupancy",
        50, eta_range_[0], eta_range_[1], "|#eta|");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2(region_id, station_id);

      me_simhit_occ_phi_[key2] = bookHist1D(
          booker, key2, "muon_simhit_occ_phi", "Muon SimHit Phi Occupancy",
          51, -M_PI, M_PI, "#phi");

      me_strip_occ_phi_[key2] = bookHist1D(
          booker, key2, "matched_strip_occ_phi", "Matched Digi Phi Occupancy",
          51, -M_PI, M_PI, "#phi");

      me_simhit_occ_det_[key2] = bookDetectorOccupancy(
          booker, key2, station, "muon_simhit", "Muon SimHit");

      me_strip_occ_det_[key2] = bookDetectorOccupancy(
          booker, key2, station, "matched_strip", "Matched Strip Digi");

    } // station looop 
  } // region loop

  // NOTE Bunch Crossing
  if (detail_plot_) {
    booker.setCurrentFolder("MuonGEMDigisV/GEMDigisTask/Strip/BunchCrossing");

    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const GEMSuperChamber* super_chamber = station->superChambers().front();
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          me_detail_bx_[key3] = bookHist1D(
              booker, key3, "strip_bx", "Strip Digi Bunch Crossing",
              5, -2.5, 2.5, "Bunch crossing");
        } // chamber loop
      } // station loop
    } // region loop
  } // detail plot
}


GEMStripDigiValidation::~GEMStripDigiValidation() {}


void GEMStripDigiValidation::analyze(const edm::Event& event,
                                     const edm::EventSetup& setup) {
  const GEMGeometry* gem = initGeometry(setup);

  edm::Handle<edm::PSimHitContainer> simhit_container;
  event.getByToken(simhit_token_, simhit_container);
  if (not simhit_container.isValid()) {
    edm::LogError(kLogCategory_) << "Failed to get PSimHitContainer." << std::endl;
    return;
  }

  edm::Handle<GEMDigiCollection> digi_collection;
  event.getByToken(strip_token_, digi_collection);
  if (not digi_collection.isValid()) {
    edm::LogError(kLogCategory_) << "Cannot get strips by Token stripToken." << std::endl;
    return;
  }

  for (const auto& simhit : *simhit_container.product()) {
    // muon only
    if (not isMuonSimHit(simhit)) {
      continue;
    }

    if (gem->idToDet(simhit.detUnitId()) == nullptr) {
      edm::LogError(kLogCategory_) << "SimHit did not match with GEMGeometry." << std::endl;
      continue;
    }

    GEMDetId simhit_gemid(simhit.detUnitId());

    Int_t region_id = simhit_gemid.region();
    Int_t station_id = simhit_gemid.station();
    Int_t layer_id = simhit_gemid.layer();
    Int_t chamber_id = simhit_gemid.chamber();
    Int_t roll_id = simhit_gemid.roll();

    ME2IdsKey key2(region_id, station_id);
    ME3IdsKey key3(region_id, station_id, layer_id);

    const GEMEtaPartition* roll = gem->etaPartition(simhit_gemid);

    const LocalPoint& simhit_local_pos = simhit.localPosition();
    const GlobalPoint& simhit_global_pos = roll->surface().toGlobal(simhit_local_pos);

    Float_t simhit_g_eta = std::abs(simhit_global_pos.eta());
    Float_t simhit_g_phi = simhit_global_pos.phi();

    Int_t simhit_strip = roll->strip(simhit_local_pos);

    Int_t bin_x = getDetOccBinX(chamber_id, layer_id);
    me_simhit_occ_eta_[region_id]->Fill(simhit_g_eta);
    me_simhit_occ_phi_[key2]->Fill(simhit_g_phi);
    me_simhit_occ_det_[key2]->Fill(bin_x, roll_id);

    Bool_t found_matched_digi = false;
    // FIXME too long
    for (auto range_iter = digi_collection->begin(); range_iter != digi_collection->end(); range_iter++) {
      if (simhit_gemid != (*range_iter).first)
        continue;

      const GEMDigiCollection::Range& range = (*range_iter).second;
      for (auto digi = range.first; digi != range.second; ++digi) {
        if (simhit_strip == digi->strip()) {
          found_matched_digi = true;

          // If we use global position of digi, 'inconsistent bin contents'
          // exception may occur.
          me_strip_occ_eta_[region_id]->Fill(simhit_g_eta);
          me_strip_occ_phi_[key2]->Fill(simhit_g_phi);
          me_strip_occ_det_[key2]->Fill(bin_x, roll_id);
          break;
        }
      }  // range loop

      if (found_matched_digi)
        break;

    } // digi_collection lop
  } // simhit_container loop
}
