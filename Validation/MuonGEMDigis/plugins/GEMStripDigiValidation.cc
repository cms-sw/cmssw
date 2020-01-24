#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonGEMDigis/plugins/GEMStripDigiValidation.h"

#include <TMath.h>
#include <iomanip>
GEMStripDigiValidation::GEMStripDigiValidation(const edm::ParameterSet& cfg) : GEMBaseValidation(cfg) {
  const auto& pset = cfg.getParameterSet("gemStripDigi");
  inputToken_ = consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("inputTag"));
}

void GEMStripDigiValidation::bookHistograms(DQMStore::IBooker& booker,
                                            edm::Run const& run,
                                            edm::EventSetup const& event_setup) {
  const GEMGeometry* gem = initGeometry(event_setup);

  // NOTE Occupancy
  const char* occ_folder = gSystem->ConcatFileName(folder_.c_str(), "Occupancy");
  booker.setCurrentFolder(occ_folder);

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    // NOTE occupancy plots for eta efficiency
    me_simhit_occ_eta_[region_id] = bookHist1D(
                                               booker, region_id, "muon_simhit_occ_eta", "Muon SimHit Eta Occupancy",
                                               50, eta_range_[0], eta_range_[1], "|#eta|");

    me_strip_occ_eta_[region_id] = bookHist1D(
                                              booker, region_id, "matched_strip_occ_eta", "Strip Digi Eta Occupancy",
                                              50, eta_range_[0], eta_range_[1], "|#eta|");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2(region_id, station_id);

      // NOTE occupancy plots for phi efficiency
      me_simhit_occ_phi_[key2] = bookHist1D(
                                            booker, key2, "muon_simhit_occ_phi", "Muon SimHit Phi Occupancy",
                                            51, -M_PI, M_PI, "#phi");

      me_strip_occ_phi_[key2] = bookHist1D(
                                           booker, key2, "matched_strip_occ_phi", "Matched Digi Phi Occupancy",
                                           51, -M_PI, M_PI, "#phi");

      // NOTE occupancy plots for detector component efficiency
      me_simhit_occ_det_[key2] = bookDetectorOccupancy(
                                                       booker, key2, station, "muon_simhit", "Muon SimHit");
      me_strip_occ_det_[key2] = bookDetectorOccupancy(
                                                      booker, key2, station, "matched_strip", "Matched Strip Digi");

    }  // End loop over station ids
  }    // End loop over region ids

  // NOTE Bunch Crossing
  if (detail_plot_) {
    const char* bx_folder = gSystem->ConcatFileName(folder_.c_str(), "BunchCrossing");
    booker.setCurrentFolder(bx_folder);

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
        }  // chamber loop
      }    // station loop
    }      // region loop
  }        // detail plot
}

GEMStripDigiValidation::~GEMStripDigiValidation() {}

void GEMStripDigiValidation::analyze(const edm::Event& event, const edm::EventSetup& event_setup) {
  const GEMGeometry* gem = initGeometry(event_setup);

  edm::Handle<edm::PSimHitContainer> simhit_container;
  event.getByToken(inputTokenSH_, simhit_container);
  if (not simhit_container.isValid()) {
    edm::LogError(log_category_) << "Failed to get PSimHitContainer." << std::endl;
    return;
  }

  edm::Handle<GEMDigiCollection> digi_collection;
  event.getByToken(inputToken_, digi_collection);
  if (not digi_collection.isValid()) {
    edm::LogError(log_category_) << "Cannot get strips by Token stripToken." << std::endl;
    return;
  }

  for (const auto& simhit : *simhit_container.product()) {
    // muon only
    if (not isMuonSimHit(simhit))
      continue;

    if (gem->idToDet(simhit.detUnitId()) == nullptr) {
      edm::LogError(log_category_) << "SimHit did not match with GEMGeometry." << std::endl;
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
      }  // end loop over range

      if (found_matched_digi)
        break;

    }  // end lopp over digi_collection
  }    // end loop over simhit_container
}
