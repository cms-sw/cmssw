#include "Validation/MuonGEMDigis/plugins/GEMCoPadDigiValidation.h"
#include <TMath.h>

GEMCoPadDigiValidation::GEMCoPadDigiValidation(const edm::ParameterSet& cfg) : GEMBaseValidation(cfg) {
  const auto& pset = cfg.getParameterSet("gemCoPadDigi");
  inputToken_ = consumes<GEMCoPadDigiCollection>(pset.getParameter<edm::InputTag>("inputTag"));
  gem_bx_min_ = pset.getParameter<int>("minBX");
  gem_bx_max_ = pset.getParameter<int>("maxBX");
}

void GEMCoPadDigiValidation::bookHistograms(DQMStore::IBooker& booker,
                                            edm::Run const& run,
                                            edm::EventSetup const& event_setup) {
  const GEMGeometry* gem = initGeometry(event_setup);

  // NOTE Occupancy
  const char* occ_folder = gSystem->ConcatFileName(folder_.c_str(), "Occupancy");
  booker.setCurrentFolder(occ_folder);

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    me_occ_zr_[region_id] = bookZROccupancy(booker, region_id, "copad", "CoPad");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      Int_t num_pads = station->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
      ME2IdsKey key2(region_id, station_id);

      me_occ_det_[key2] = bookDetectorOccupancy(booker, key2, station, "copad", "CoPad");

      if (detail_plot_) {
        me_detail_occ_xy_[key2] = bookXYOccupancy(booker, key2, "copad","CoPad Digi");


        me_detail_occ_phi_pad_[key2] = bookHist2D(
                                                  booker, key2,
                                                  "copad_digi_occ_phi_pad",
                                                  "CoPad Digi Occupancy",
                                                  280, -M_PI, M_PI,
                                                  num_pads / 2, 0, num_pads,
                                                  "#phi [rad]", "Pad number");

        me_detail_occ_pad_[key2] = bookHist1D(
                                              booker, key2,
                                              "copad_digi_occ_pad",
                                              "CoPad Digi Ocupancy per pad number",
                                              num_pads, 0.5, num_pads + 0.5,
                                              "Pad number");
      }
    }  // end loop over station ids
  }    // end loop over region ids

  // NOTE Bunch Crossing
  if (detail_plot_) {
    const char* bx_folder = gSystem->ConcatFileName(folder_.c_str(), "BunchCrossing");
    booker.setCurrentFolder(bx_folder);

    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();
        ME2IdsKey key2(region_id, station_id);
        me_detail_bx_[key2] = bookHist1D(
                                         booker, key2,
                                         "copad_digi_bx",
                                         "CoPad Digi Bunch Crossing",
                                         5, -2.5, 2.5,
                                         "Bunch crossing");
      }  // station loop
    }    // region loop
  }      // detail plot
}

GEMCoPadDigiValidation::~GEMCoPadDigiValidation() {}

void GEMCoPadDigiValidation::analyze(const edm::Event& event, const edm::EventSetup& event_setup) {
  const GEMGeometry* gem = initGeometry(event_setup);

  edm::Handle<GEMCoPadDigiCollection> copad_collection;
  event.getByToken(inputToken_, copad_collection);
  if (not copad_collection.isValid()) {
    edm::LogError(log_category_) << "Cannot get pads by token.";
    return;
  }

  for (auto range_iter = copad_collection->begin(); range_iter != copad_collection->end(); range_iter++) {
    GEMDetId gemid = (*range_iter).first;
    const GEMCoPadDigiCollection::Range& range = (*range_iter).second;

    Int_t region_id = gemid.region();
    Int_t station_id = gemid.station();
    Int_t ring_id = gemid.ring();
    Int_t layer_id = gemid.layer();
    Int_t chamber_id = gemid.chamber();

    ME2IdsKey key2(region_id, station_id);

    for (auto digi = range.first; digi != range.second; ++digi) {
      // GEM copads are stored per super chamber!
      // layer_id = 0, roll_id = 0
      GEMDetId super_chamber_id = GEMDetId(region_id, ring_id, station_id, 0, chamber_id, 0);
      Int_t roll_id = (*digi).roll();

      const GeomDet* geom_det = gem->idToDet(super_chamber_id);
      if (geom_det == nullptr) {
        edm::LogError(log_category_) << super_chamber_id << " : This detId cannot be "
                                     << "loaded from GEMGeometry // Original" << gemid << " station : " << station_id
                                     << std::endl
                                     << "Getting DetId failed. Discard this gem copad hit." << std::endl;
        continue;
      }

      const BoundPlane& surface = geom_det->surface();
      const GEMSuperChamber* superChamber = gem->superChamber(super_chamber_id);

      Int_t pad1 = digi->pad(1);
      Int_t pad2 = digi->pad(2);
      Int_t bx1 = digi->bx(1);
      Int_t bx2 = digi->bx(2);

      // Filtered using BX
      if (bx1 < gem_bx_min_ or bx1 > gem_bx_max_)
        continue;
      if (bx2 < gem_bx_min_ or bx2 > gem_bx_max_)
        continue;

      const LocalPoint& lp1 = superChamber->chamber(1)->etaPartition(roll_id)->centreOfPad(pad1);
      const LocalPoint& lp2 = superChamber->chamber(2)->etaPartition(roll_id)->centreOfPad(pad2);

      const GlobalPoint& gp1 = surface.toGlobal(lp1);
      const GlobalPoint& gp2 = surface.toGlobal(lp2);

      Float_t g_r1 = gp1.perp();
      Float_t g_r2 = gp2.perp();

      Float_t g_z1 = gp1.z();
      Float_t g_z2 = gp2.z();

      Float_t g_phi = gp1.phi();
      Float_t g_x = gp1.x();
      Float_t g_y = gp1.y();

      // Fill normal plots.
      me_occ_zr_[region_id]->Fill(std::fabs(g_z1), g_r1);
      me_occ_zr_[region_id]->Fill(std::fabs(g_z2), g_r2);

      Int_t bin_x = getDetOccBinX(chamber_id, layer_id);
      me_occ_det_[key2]->Fill(bin_x, roll_id);
      me_occ_det_[key2]->Fill(bin_x + 1, roll_id);

      // Fill detail plots.
      if (detail_plot_) {
        me_detail_occ_xy_[key2]->Fill(g_x, g_y);
        me_detail_occ_phi_pad_[key2]->Fill(g_phi, pad1);
        me_detail_occ_pad_[key2]->Fill(pad1);
        me_detail_bx_[key2]->Fill(bx1);
        me_detail_bx_[key2]->Fill(bx2);
      } // detailPlot_
    }
  }
}
