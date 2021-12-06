#include "Validation/MuonGEMDigis/plugins/GEMCoPadDigiValidation.h"

GEMCoPadDigiValidation::GEMCoPadDigiValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMCoPadDigiValidation") {
  const auto& copad_pset = pset.getParameterSet("gemCoPadDigi");

  const auto& copad_tag = copad_pset.getParameter<edm::InputTag>("inputTag");
  copad_token_ = consumes<GEMCoPadDigiCollection>(copad_tag);
  geomToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  geomTokenBeginRun_ = esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();

  gem_bx_min_ = copad_pset.getParameter<int>("minBX");
  gem_bx_max_ = copad_pset.getParameter<int>("maxBX");
}

void GEMCoPadDigiValidation::bookHistograms(DQMStore::IBooker& booker,
                                            edm::Run const& run,
                                            edm::EventSetup const& setup) {
  const GEMGeometry* gem = &setup.getData(geomTokenBeginRun_);

  // NOTE Occupancy
  booker.setCurrentFolder("GEM/CoPad");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    if (detail_plot_)
      me_detail_occ_zr_[region_id] = bookZROccupancy(booker, region_id, "copad", "CoPad");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      const auto& superChamberVec = station->superChambers();
      if (superChamberVec.empty() || superChamberVec[0] == nullptr) {
        edm::LogError(kLogCategory_) << "Super chambers missing or null for region = " << region_id
                                     << " and station = " << station_id;
        continue;
      }
      const auto& chamberVec = superChamberVec[0]->chambers();
      if (chamberVec.empty() || chamberVec[0] == nullptr) {
        edm::LogError(kLogCategory_) << "Chambers missing or null for region, station, super chamber = (" << region_id
                                     << ", " << station_id << ", " << superChamberVec[0]->id() << ")";
        continue;
      }
      const auto& etaPartitionVec = chamberVec[0]->etaPartitions();
      if (etaPartitionVec.empty() || etaPartitionVec[0] == nullptr) {
        edm::LogError(kLogCategory_) << "Eta partition missing or null for region, station, super chamber, chamber = ("
                                     << region_id << ", " << station_id << ", " << superChamberVec[0]->id() << ", "
                                     << chamberVec[0]->id() << ")";
        continue;
      }
      Int_t num_pads = etaPartitionVec[0]->npads();
      ME2IdsKey key2{region_id, station_id};

      if (detail_plot_) {
        me_detail_occ_xy_[key2] = bookXYOccupancy(booker, key2, "copad", "CoPad");

        me_detail_occ_phi_pad_[key2] = bookHist2D(booker,
                                                  key2,
                                                  "copad_occ_phi_pad",
                                                  "CoPad Occupancy",
                                                  280,
                                                  -M_PI,
                                                  M_PI,
                                                  num_pads / 2,
                                                  0,
                                                  num_pads,
                                                  "#phi [rad]",
                                                  "Pad number");

        me_detail_occ_pad_[key2] = bookHist1D(
            booker, key2, "copad_occ_pad", "CoPad Ocupancy per pad number", num_pads, 0.5, num_pads + 0.5, "Pad number");

        me_detail_occ_det_[key2] = bookDetectorOccupancy(booker, key2, station, "copad", "CoPad");
      }
    }  // station loop
  }    // region loop

  // NOTE Bunch Crossing
  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();
        ME2IdsKey key2{region_id, station_id};

        me_detail_bx_[key2] =
            bookHist1D(booker, key2, "copad_bx", "CoPad Bunch Crossing", 5, -2.5, 2.5, "Bunch crossing");
      }  // station loop
    }    // region loop
  }      // detail plot
}

GEMCoPadDigiValidation::~GEMCoPadDigiValidation() {}

void GEMCoPadDigiValidation::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  const GEMGeometry* gem = &setup.getData(geomToken_);

  edm::Handle<GEMCoPadDigiCollection> copad_collection;
  event.getByToken(copad_token_, copad_collection);
  if (not copad_collection.isValid()) {
    edm::LogError(kLogCategory_) << "Cannot get pads by token." << std::endl;
    return;
  }

  // GEMCoPadDigiCollection::DigiRangeIterator
  for (const auto& copad_pair : *copad_collection) {
    GEMDetId gemid = copad_pair.first;
    const auto& range = copad_pair.second;

    Int_t region_id = gemid.region();
    Int_t station_id = gemid.station();
    Int_t ring_id = gemid.ring();
    Int_t layer_id = gemid.layer();
    Int_t chamber_id = gemid.chamber();
    Int_t num_layers = gemid.nlayers();

    ME2IdsKey key2{region_id, station_id};

    for (auto digi = range.first; digi != range.second; ++digi) {
      // GEM copads are stored per super chamber!
      // layer_id = 0, roll_id = 0
      GEMDetId super_chamber_id{region_id, ring_id, station_id, 0, chamber_id, 0};
      Int_t roll_id = (*digi).roll();

      const GeomDet* geom_det = gem->idToDet(super_chamber_id);
      if (geom_det == nullptr) {
        edm::LogError(kLogCategory_) << super_chamber_id << " : This detId cannot be "
                                     << "loaded from GEMGeometry // Original" << gemid << " station : " << station_id
                                     << std::endl
                                     << "Getting DetId failed. Discard this gem copad hit." << std::endl;
        continue;
      }

      const BoundPlane& surface = geom_det->surface();
      const GEMSuperChamber* super_chamber = gem->superChamber(super_chamber_id);
      if (super_chamber == nullptr) {
        edm::LogError(kLogCategory_) << "Super chamber is null for super chamber ID = " << super_chamber_id;
        continue;
      }
      Int_t pad1 = digi->pad(1);
      Int_t pad2 = digi->pad(2);
      Int_t bx1 = digi->bx(1);
      Int_t bx2 = digi->bx(2);

      // Filtered using BX
      if (bx1 < gem_bx_min_ or bx1 > gem_bx_max_)
        continue;
      if (bx2 < gem_bx_min_ or bx2 > gem_bx_max_)
        continue;

      const Int_t padArray[] = {pad1, pad2};
      LocalPoint lpArray[2];
      int ptCounter = 0;
      for (; ptCounter < 2; ++ptCounter) {
        const GEMChamber* const chamber = super_chamber->chamber(ptCounter + 1);  // Fetch chambers 1 and 2
        if (chamber == nullptr) {
          edm::LogError(kLogCategory_) << "Chamber " << (ptCounter + 1)
                                       << " is null for super chamber = " << super_chamber_id;
          break;
        }
        const GEMEtaPartition* const etaPartition = chamber->etaPartition(roll_id);
        if (etaPartition == nullptr) {
          edm::LogError(kLogCategory_) << "Eta partition " << roll_id << " is null for chamber, super chamber = ("
                                       << (ptCounter + 1) << ", " << super_chamber_id << ")";
          break;
        }
        lpArray[ptCounter] = etaPartition->centreOfPad(padArray[ptCounter]);
      }                     // end for
      if (ptCounter < 2) {  // Broke out of "for" loop
        edm::LogError(kLogCategory_) << "Skipping a digi due to bad chamber " << (ptCounter + 1);
        continue;
      }
      const GlobalPoint& gp1 = surface.toGlobal(lpArray[0]);
      const GlobalPoint& gp2 = surface.toGlobal(lpArray[1]);

      Float_t g_r1 = gp1.perp();
      Float_t g_r2 = gp2.perp();

      Float_t g_z1 = gp1.z();
      Float_t g_z2 = gp2.z();

      Float_t g_phi = gp1.phi();
      Float_t g_x = gp1.x();
      Float_t g_y = gp1.y();

      // Fill normal plots.

      Int_t bin_x = getDetOccBinX(num_layers, chamber_id, layer_id);

      // Fill detail plots.
      if (detail_plot_) {
        me_detail_occ_zr_[region_id]->Fill(std::fabs(g_z1), g_r1);
        me_detail_occ_zr_[region_id]->Fill(std::fabs(g_z2), g_r2);
        me_detail_occ_xy_[key2]->Fill(g_x, g_y);
        me_detail_occ_det_[key2]->Fill(bin_x, roll_id);
        me_detail_occ_det_[key2]->Fill(bin_x + 1, roll_id);
        me_detail_occ_phi_pad_[key2]->Fill(g_phi, pad1);
        me_detail_occ_pad_[key2]->Fill(pad1);
        me_detail_bx_[key2]->Fill(bx1);
        me_detail_bx_[key2]->Fill(bx2);
      }  // detail_plot_
    }    // loop over digis
  }      // loop over range iters
}
