#include "DataFormats/Common/interface/Handle.h"
#include "Validation/MuonME0Validation/interface/ME0HitsValidation.h"
#include <TMath.h>

ME0HitsValidation::ME0HitsValidation(const edm::ParameterSet &cfg) : ME0BaseValidation(cfg) {
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
}

void ME0HitsValidation::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &Run, edm::EventSetup const &iSetup) {
  LogDebug("MuonME0HitsValidation") << "Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0HitsV/ME0HitsTask");

  unsigned int nregion = 2;

  edm::LogInfo("MuonME0HitsValidation") << "+++ Info : # of region : " << nregion << std::endl;

  LogDebug("MuonME0HitsValidation") << "+++ Info : finish to get geometry information from ES.\n";

  for (unsigned int region_num = 0; region_num < nregion; region_num++) {
    me0_sh_tot_zr[region_num] = BookHistZR(ibooker, "me0_sh", "SimHit", region_num);
    for (unsigned int layer_num = 0; layer_num < 6; layer_num++) {
      me0_sh_zr[region_num][layer_num] = BookHistZR(ibooker, "me0_sh", "SimHit", region_num, layer_num);
      me0_sh_xy[region_num][layer_num] = BookHistXY(ibooker, "me0_sh", "SimHit", region_num, layer_num);
      std::string hist_name_for_tof =
          std::string("me0_sh_tof_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_name_for_tofMu =
          std::string("me0_sh_tofMuon_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_name_for_eloss =
          std::string("me0_sh_energyloss_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_name_for_elossMu =
          std::string("me0_sh_energylossMuon_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_label_for_xy = "SimHit occupancy : region" + regionLabel[region_num] + " layer " +
                                      layerLabel[layer_num] + " ; globalX [cm]; globalY[cm]";
      std::string hist_label_for_tof = "SimHit TOF : region" + regionLabel[region_num] + " layer " +
                                       layerLabel[layer_num] + " " + " ; Time of flight [ns] ; entries";
      std::string hist_label_for_tofMu = "SimHit TOF(Muon only) : region" + regionLabel[region_num] + " layer " +
                                         layerLabel[layer_num] + " " + " ; Time of flight [ns] ; entries";
      std::string hist_label_for_eloss = "SimHit energy loss : region" + regionLabel[region_num] + " layer " +
                                         layerLabel[layer_num] + " " + " ; Energy loss [eV] ; entries";
      std::string hist_label_for_elossMu = "SimHit energy loss(Muon only) : region" + regionLabel[region_num] +
                                           " layer " + layerLabel[layer_num] + " " + " ; Energy loss [eV] ; entries";

      double tof_min, tof_max;
      tof_min = 10;
      tof_max = 30;
      me0_sh_tof[region_num][layer_num] =
          ibooker.book1D(hist_name_for_tof.c_str(), hist_label_for_tof.c_str(), 40, tof_min, tof_max);
      me0_sh_tofMu[region_num][layer_num] =
          ibooker.book1D(hist_name_for_tofMu.c_str(), hist_label_for_tofMu.c_str(), 40, tof_min, tof_max);
      me0_sh_eloss[region_num][layer_num] =
          ibooker.book1D(hist_name_for_eloss.c_str(), hist_label_for_eloss.c_str(), 60, 0., 6000.);
      me0_sh_elossMu[region_num][layer_num] =
          ibooker.book1D(hist_name_for_elossMu.c_str(), hist_label_for_elossMu.c_str(), 60, 0., 6000.);
    }
  }
}

ME0HitsValidation::~ME0HitsValidation() {}

void ME0HitsValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  edm::ESHandle<ME0Geometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const ME0Geometry *ME0Geometry_ = (&*hGeom);

  edm::Handle<edm::PSimHitContainer> ME0Hits;
  e.getByToken(InputTagToken_, ME0Hits);
  if (!ME0Hits.isValid()) {
    edm::LogError("ME0HitsValidation") << "Cannot get ME0Hits by Token simInputTagToken";
    return;
  }

  Float_t timeOfFlightMuon = 0.;
  Float_t energyLossMuon = 0;

  for (auto hits = ME0Hits->begin(); hits != ME0Hits->end(); hits++) {
    const ME0DetId id(hits->detUnitId());
    Int_t region = id.region();
    Int_t layer = id.layer();

    // Int_t even_odd = id.chamber()%2;
    if (ME0Geometry_->idToDet(hits->detUnitId()) == nullptr) {
      std::cout << "simHit did not matched with GEMGeometry." << std::endl;
      continue;
    }

    const LocalPoint hitLP(hits->localPosition());

    const GlobalPoint hitGP(ME0Geometry_->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
    Float_t g_r = hitGP.perp();
    Float_t g_x = hitGP.x();
    Float_t g_y = hitGP.y();
    Float_t g_z = hitGP.z();
    Float_t energyLoss = hits->energyLoss();
    Float_t timeOfFlight = hits->timeOfFlight();

    if (abs(hits->particleType()) == 13) {
      timeOfFlightMuon = hits->timeOfFlight();
      energyLossMuon = hits->energyLoss();
      // fill histos for Muons only
      me0_sh_tofMu[(int)(region / 2. + 0.5)][layer - 1]->Fill(timeOfFlightMuon);
      me0_sh_elossMu[(int)(region / 2. + 0.5)][layer - 1]->Fill(energyLossMuon * 1.e9);
    }

    me0_sh_zr[(int)(region / 2. + 0.5)][layer - 1]->Fill(g_z, g_r);
    me0_sh_tot_zr[(int)(region / 2. + 0.5)]->Fill(g_z, g_r);
    me0_sh_xy[(int)(region / 2. + 0.5)][layer - 1]->Fill(g_x, g_y);
    me0_sh_tof[(int)(region / 2. + 0.5)][layer - 1]->Fill(timeOfFlight);
    me0_sh_eloss[(int)(region / 2. + 0.5)][layer - 1]->Fill(energyLoss * 1.e9);
  }
}
