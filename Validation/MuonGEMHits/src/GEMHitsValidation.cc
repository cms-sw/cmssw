#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include <exception>
using namespace std;
GEMHitsValidation::GEMHitsValidation(const edm::ParameterSet& cfg) : GEMBaseValidation(cfg) {
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  detailPlot_ = cfg.getParameter<bool>("detailPlot");
}

void GEMHitsValidation::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& Run, edm::EventSetup const& iSetup) {
  const GEMGeometry* GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr) {
    std::cout << "geometry is wrong! Terminated." << std::endl;
    return;
  }

  ibooker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask");
  edm::LogInfo("MuonGEMHitsValidation") << "+++ Info : # of region : " << nRegion() << std::endl;
  edm::LogInfo("MuonGEMHitsValidation") << "+++ Info : # of stations : " << nStation() << std::endl;
  edm::LogInfo("MuonGEMHitsValidation") << "+++ Info : # of eta partition : " << nPart() << std::endl;

  LogDebug("MuonGEMHitsValidation") << "+++ Info : finish to get geometry information from ES.\n";

  LogDebug("MuonGEMHitsValidation") << "+++ Region independant part.\n";
  // Region independant.
  for (auto& station : GEMGeometry_->regions()[0]->stations()) {
    int st = station->station();
    // TOF and Energy loss part are indepent from Region.
    // Labeling TOF and Energy loss
    TString hist_name_for_tofMu = TString::Format("gem_sh_simple_tofMuon_st%s", getStationLabel(st).c_str());
    TString hist_name_for_elossMu = TString::Format("gem_sh_simple_energylossMuon_st%s", getStationLabel(st).c_str());
    TString hist_label_for_tofMu = TString::Format(
        "SimHit TOF(Muon only) station : station %s ; Time of flight [ns] ; entries", getStationLabel(st).c_str());
    TString hist_label_for_elossMu = TString::Format(
        "SimHit energy loss(Muon only) : station %s ; Energy loss [eV] ; entries", getStationLabel(st).c_str());
    // set tof's range.
    double tof_min, tof_max;
    if (st == 1) {
      tof_min = 18;
      tof_max = 22;
    } else {
      tof_min = 26;
      tof_max = 30;
    }
    gem_sh_simple_tofMu[hist_name_for_tofMu.Hash()] =
        ibooker.book1D(hist_name_for_tofMu.Data(), hist_label_for_tofMu.Data(), 40, tof_min, tof_max);
    gem_sh_simple_elossMu[hist_name_for_elossMu.Hash()] =
        ibooker.book1D(hist_name_for_elossMu.Data(), hist_label_for_elossMu.Data(), 60, 0., 6000.);
  }

  LogDebug("MuonGEMHitsValidation") << "+++ Region+Station part.\n";
  // Regions, Region+station
  for (auto& region : GEMGeometry_->regions()) {
    int re = region->region();
    TString title_suffix = getSuffixTitle(re);
    TString histname_suffix = getSuffixName(re);
    LogDebug("MuonGEMHitsValidation") << "+++ SimpleZR Occupancy\n";
    TString simpleZR_title = TString::Format("ZR Occupancy%s; |Z|(cm); R(cm)", title_suffix.Data());
    TString simpleZR_histname = TString::Format("hit_simple_zr%s", histname_suffix.Data());

    MonitorElement* simpleZR = getSimpleZR(ibooker, simpleZR_title, simpleZR_histname);
    if (simpleZR != nullptr) {
      Hit_simple_zr[simpleZR_histname.Hash()] = simpleZR;
    }

    for (auto& station : region->stations()) {
      int st = station->station();
      TString title_suffix2 = getSuffixTitle(re, st);
      TString histname_suffix2 = getSuffixName(re, st);
      LogDebug("MuonGEMHitsValidation") << "+++ dcEta Occupancy\n";
      TString dcEta_title =
          TString::Format("Occupancy for detector component %s;;#eta-partition", title_suffix2.Data());
      TString dcEta_histname = TString::Format("hit_dcEta%s", histname_suffix2.Data());
      MonitorElement* dcEta = getDCEta(ibooker, station, dcEta_title, dcEta_histname);
      if (dcEta != nullptr) {
        Hit_dcEta[dcEta_histname.Hash()] = dcEta;
      }
    }
  }

  LogDebug("MuonGEMHitsValidation") << "+++ Begining Detail Plots\n";
  if (detailPlot_) {
    for (auto& region : GEMGeometry_->regions()) {
      for (auto& station : region->stations()) {
        for (auto& ring : station->rings()) {
          if (ring->ring() != 1)
            break;  // Only Ring1 is interesting.
          string name_suffix = getSuffixName(region->region(), station->station());
          string title_suffix = getSuffixTitle(region->region(), station->station());

          TString hist_name = TString::Format("gem_sh_xy%s", name_suffix.c_str());
          TString hist_title = TString::Format("Simhit Global XY Plots at %s", title_suffix.c_str());
          MonitorElement* temp = ibooker.book2D(
              (hist_name + "_even").Data(), (hist_title + " even").Data(), nBinXY_, -360, 360, nBinXY_, -360, 360);
          if (temp != nullptr) {
            LogDebug("MuonGEMHitsValidation") << "ME can be acquired!";
          } else {
            LogDebug("MuonGEMHitsValidation") << "ME can not be acquired!";
            return;
          }
          gem_sh_xy_st_ch[(hist_name + "_even").Hash()] = temp;

          MonitorElement* temp2 = ibooker.book2D(
              (hist_name + "_odd").Data(), (hist_title + " odd").Data(), nBinXY_, -360, 360, nBinXY_, -360, 360);
          if (temp2 != nullptr) {
            LogDebug("MuonGEMHitsValidation") << "ME can be acquired!";
          } else {
            LogDebug("MuonGEMHitsValidation") << "ME can not be acquired!";
            return;
          }
          gem_sh_xy_st_ch[(hist_name + "_odd").Hash()] = temp2;
        }
      }
    }
    for (unsigned int region_num = 0; region_num < nRegion(); region_num++) {
      for (unsigned int station_num = 0; station_num < nStation(); station_num++) {
        for (unsigned int layer_num = 0; layer_num < 2; layer_num++) {
          gem_sh_zr[region_num][station_num][layer_num] =
              BookHistZR(ibooker, "gem_sh", "SimHit", region_num, station_num, layer_num);
          gem_sh_xy[region_num][station_num][layer_num] =
              BookHistXY(ibooker, "gem_sh", "SimHit", region_num, station_num, layer_num);

          int re = ((unsigned int)region_num) * 2 - 1;
          std::string suffixname = getSuffixName(re, station_num + 1, layer_num + 1);
          std::string suffixtitle = getSuffixTitle(re, station_num + 1, layer_num + 1);
          std::string hist_name_for_tof = std::string("gem_sh_tof_") + suffixname;
          std::string hist_name_for_tofMu = std::string("gem_sh_tofMuon_") + suffixname;
          std::string hist_name_for_eloss = std::string("gem_sh_energyloss_") + suffixname;
          std::string hist_name_for_elossMu = std::string("gem_sh_energylossMuon_") + suffixname;
          std::string hist_label_for_xy = "SimHit occupancy : region" + suffixtitle + " ; globalX [cm]; globalY[cm]";
          std::string hist_label_for_tof = "SimHit TOF : region" + suffixtitle + " ; Time of flight [ns] ; entries";
          std::string hist_label_for_tofMu =
              "SimHit TOF(Muon only) : " + suffixtitle + " ; Time of flight [ns] ; entries";
          std::string hist_label_for_eloss = "SimHit energy loss : " + suffixtitle + " ; Energy loss [eV] ; entries";
          std::string hist_label_for_elossMu =
              "SimHit energy loss(Muon only) : " + suffixtitle + " ; Energy loss [eV] ; entries";

          double tof_min, tof_max;
          if (station_num == 0) {
            tof_min = 18;
            tof_max = 22;
          } else {
            tof_min = 26;
            tof_max = 30;
          }
          gem_sh_tof[region_num][station_num][layer_num] =
              ibooker.book1D(hist_name_for_tof.c_str(), hist_label_for_tof.c_str(), 40, tof_min, tof_max);
          gem_sh_tofMu[region_num][station_num][layer_num] =
              ibooker.book1D(hist_name_for_tofMu.c_str(), hist_label_for_tofMu.c_str(), 40, tof_min, tof_max);
          gem_sh_eloss[region_num][station_num][layer_num] =
              ibooker.book1D(hist_name_for_eloss.c_str(), hist_label_for_eloss.c_str(), 60, 0., 6000.);
          gem_sh_elossMu[region_num][station_num][layer_num] =
              ibooker.book1D(hist_name_for_elossMu.c_str(), hist_label_for_elossMu.c_str(), 60, 0., 6000.);
        }
      }
    }
  }
}

GEMHitsValidation::~GEMHitsValidation() {}

void GEMHitsValidation::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const GEMGeometry* GEMGeometry_ = initGeometry(iSetup);

  edm::Handle<edm::PSimHitContainer> GEMHits;
  e.getByToken(InputTagToken_, GEMHits);
  if (!GEMHits.isValid()) {
    edm::LogError("GEMHitsValidation") << "Cannot get GEMHits by Token simInputTagToken";
    return;
  }

  for (auto hits = GEMHits->begin(); hits != GEMHits->end(); hits++) {
    const GEMDetId id(hits->detUnitId());
    Int_t region = (Int_t)id.region();
    Int_t station = (Int_t)id.station();
    Int_t layer = (Int_t)id.layer();
    Int_t chamber = (Int_t)id.chamber();
    Int_t nroll = (Int_t)id.roll();

    //Int_t even_odd = id.chamber()%2;
    if (GEMGeometry_->idToDet(GEMDetId(hits->detUnitId())) == nullptr) {
      std::cout << "simHit did not matched with GEMGeometry." << std::endl;
      continue;
    }
    //const LocalPoint p0(0., 0., 0.);
    //const GlobalPoint Gp0(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(p0));
    const LocalPoint hitLP(hits->localPosition());

    const GlobalPoint hitGP(GEMGeometry_->idToDet(GEMDetId(hits->detUnitId()))->surface().toGlobal(hitLP));
    Float_t g_r = hitGP.perp();
    Float_t g_x = hitGP.x();
    Float_t g_y = hitGP.y();
    Float_t g_z = hitGP.z();
    Float_t energyLoss = hits->energyLoss();
    Float_t timeOfFlight = hits->timeOfFlight();

    int layer_num = layer - 1;
    int binX = (chamber - 1) * 2 + layer_num;
    int binY = nroll;

    //const LocalPoint hitEP(hits->entryPoint());

    TString histname_suffix = getSuffixName(region);
    TString simple_zr_histname = TString::Format("hit_simple_zr%s", histname_suffix.Data());
    LogDebug("GEMHitsValidation") << simple_zr_histname << std::endl;
    Hit_simple_zr[simple_zr_histname.Hash()]->Fill(fabs(g_z), g_r);

    histname_suffix = getSuffixName(region, station);
    TString dcEta_histname = TString::Format("hit_dcEta%s", histname_suffix.Data());
    LogDebug("GEMHitsValidation") << dcEta_histname << std::endl;
    Hit_dcEta[dcEta_histname.Hash()]->Fill(binX, binY);

    TString tofMu = TString::Format("gem_sh_simple_tofMuon_st%s", getStationLabel(station).c_str());
    TString elossMu = TString::Format("gem_sh_simple_energylossMuon_st%s", getStationLabel(station).c_str());

    if (abs(hits->particleType()) == 13) {
      LogDebug("GEMHitsValidation") << tofMu << std::endl;
      gem_sh_simple_tofMu[tofMu.Hash()]->Fill(timeOfFlight);
      LogDebug("GEMHitsValidation") << elossMu << std::endl;
      gem_sh_simple_elossMu[elossMu.Hash()]->Fill(energyLoss * 1.e9);
    }

    if (detailPlot_) {
      // First, fill variable has no condition.
      LogDebug("GEMHitsValidation") << "gzgr" << std::endl;
      gem_sh_zr[(int)(region / 2. + 0.5)][station - 1][layer_num]->Fill(g_z, g_r);
      LogDebug("GEMHitsValidation") << "gxgy" << std::endl;
      gem_sh_xy[(int)(region / 2. + 0.5)][station - 1][layer_num]->Fill(g_x, g_y);
      gem_sh_tof[(int)(region / 2. + 0.5)][station - 1][layer_num]->Fill(timeOfFlight);
      gem_sh_eloss[(int)(region / 2. + 0.5)][station - 1][layer_num]->Fill(energyLoss * 1.e9);
      if (abs(hits->particleType()) == 13) {
        gem_sh_tofMu[(int)(region / 2. + 0.5)][station - 1][layer_num]->Fill(timeOfFlight);
        gem_sh_elossMu[(int)(region / 2. + 0.5)][station - 1][layer_num]->Fill(energyLoss * 1.e9);
      }
      std::string chamber = "";
      if (id.chamber() % 2 == 1)
        chamber = "odd";
      else
        chamber = "even";
      TString hist_name = TString::Format("gem_sh_xy%s", (getSuffixName(id.region(), station) + "_" + chamber).c_str());

      LogDebug("GEMHitsValidation") << hist_name << std::endl;
      gem_sh_xy_st_ch[hist_name.Hash()]->Fill(g_x, g_y);
    }
  }
}
