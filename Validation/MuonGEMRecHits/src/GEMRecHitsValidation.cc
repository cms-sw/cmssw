#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Validation/MuonGEMRecHits/interface/GEMRecHitsValidation.h"
#include <iomanip>

using namespace std;

GEMRecHitsValidation::GEMRecHitsValidation(const edm::ParameterSet &cfg) : GEMBaseValidation(cfg) {
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_RH = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));
  detailPlot_ = cfg.getParameter<bool>("detailPlot");
}

MonitorElement *GEMRecHitsValidation::BookHist1D(DQMStore::IBooker &ibooker,
                                                 const char *name,
                                                 const char *label,
                                                 unsigned int region_num,
                                                 unsigned int station_num,
                                                 unsigned int layer_num,
                                                 const unsigned int Nbin,
                                                 const Float_t xMin,
                                                 const Float_t xMax) {
  string hist_name = name + getSuffixName(region_num, station_num + 1, layer_num + 1);
  string hist_label = label + string(" : ") + getSuffixTitle(region_num, station_num + 1, layer_num + 1);
  return ibooker.book1D(hist_name, hist_label, Nbin, xMin, xMax);
}

MonitorElement *GEMRecHitsValidation::BookHist1D(DQMStore::IBooker &ibooker,
                                                 const char *name,
                                                 const char *label,
                                                 unsigned int region_num,
                                                 const unsigned int Nbin,
                                                 const Float_t xMin,
                                                 const Float_t xMax) {
  string hist_name = name + getSuffixName(region_num);
  string hist_label = label + string(" : ") + getSuffixName(region_num);
  return ibooker.book1D(hist_name, hist_label, Nbin, xMin, xMax);
}

void GEMRecHitsValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                          edm::Run const &Run,
                                          edm::EventSetup const &iSetup) {
  const GEMGeometry *GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;

  LogDebug("GEMRecHitsValidation") << "Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask");
  LogDebug("GEMRecHitsValidation") << "ibooker set current folder\n";

  gem_cls_tot = ibooker.book1D("gem_cls_tot", "ClusterSize Distribution", 11, -0.5, 10.5);
  for (auto &region : GEMGeometry_->regions()) {
    int re = region->region();
    TString title_suffix = getSuffixTitle(re);
    TString histname_suffix = getSuffixName(re);
    TString simpleZR_title = TString::Format("ZR Occupancy%s; |Z|(cm) ; R(cm)", title_suffix.Data());
    TString simpleZR_histname = TString::Format("rh_simple_zr%s", histname_suffix.Data());
    auto *simpleZR = getSimpleZR(ibooker, simpleZR_title, simpleZR_histname);
    if (simpleZR != nullptr) {
      recHits_simple_zr[simpleZR_histname.Hash()] = simpleZR;
    }

    for (auto &station : region->stations()) {
      int station_num = station->station();
      TString title_suffix2 = title_suffix + TString::Format("  Station%d", station_num);
      TString histname_suffix2 = histname_suffix + TString::Format("_st%d", station_num);

      TString dcEta_title =
          TString::Format("Occupancy for detector component %s;;#eta-partition", title_suffix2.Data());
      TString dcEta_histname = TString::Format("rh_dcEta%s", histname_suffix2.Data());
      auto *dcEta = getDCEta(ibooker, station, dcEta_title, dcEta_histname);
      if (dcEta != nullptr) {
        recHits_dcEta[dcEta_histname.Hash()] = dcEta;
      }
      int idx = 0;
      for (unsigned int sCh = 1; sCh <= station->superChambers().size(); sCh++) {
        for (unsigned int Ch = 1; Ch <= 2; Ch++) {
          idx++;
          TString label = TString::Format("ch%d_la%d", sCh, Ch);
          recHits_dcEta[dcEta_histname.Hash()]->setBinLabel(idx, label.Data());
        }
      }
    }
  }

  for (unsigned int region_num = 0; region_num < nRegion(); region_num++) {
    gem_region_pullX[region_num] = BookHist1D(ibooker, "pullX", "Pull Of X", region_num, 100, -50, 50);
    gem_region_pullY[region_num] = BookHist1D(ibooker, "pullY", "Pull Of Y", region_num, 100, -50, 50);
  }

  if (detailPlot_) {
    for (unsigned int region_num = 0; region_num < nRegion(); region_num++) {
      for (int layer_num = 0; layer_num < 2; layer_num++) {
        for (unsigned int station_num = 0; station_num < nStation(); station_num++) {
          gem_cls[region_num][station_num][layer_num] = BookHist1D(
              ibooker, "cls", "ClusterSize Distribution", region_num, station_num, layer_num, 11, -0.5, 10.5);
          gem_pullX[region_num][station_num][layer_num] =
              BookHist1D(ibooker, "pullX", "Pull Of X", region_num, station_num, layer_num, 100, -50, 50);
          gem_pullY[region_num][station_num][layer_num] =
              BookHist1D(ibooker, "pullY", "Pull Of Y", region_num, station_num, layer_num, 100, -50, 50);
          gem_rh_zr[region_num][station_num][layer_num] =
              BookHistZR(ibooker, "rh", "RecHits", region_num, station_num, layer_num);
          gem_rh_xy[region_num][station_num][layer_num] =
              BookHistXY(ibooker, "rh", "RecHits", region_num, station_num, layer_num);
        }
      }
    }
  }
  LogDebug("GEMRecHitsValidation") << "Booking End.\n";
}

GEMRecHitsValidation::~GEMRecHitsValidation() {}

void GEMRecHitsValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const GEMGeometry *GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;

  edm::Handle<GEMRecHitCollection> gemRecHits;
  edm::Handle<edm::PSimHitContainer> gemSimHits;
  e.getByToken(this->InputTagToken_, gemSimHits);
  e.getByToken(this->InputTagToken_RH, gemRecHits);
  if (!gemRecHits.isValid()) {
    edm::LogError("GEMRecHitsValidation") << "Cannot get strips by Token RecHits Token.\n";
    return;
  }

  for (edm::PSimHitContainer::const_iterator hits = gemSimHits->begin(); hits != gemSimHits->end(); ++hits) {
    const GEMDetId id(hits->detUnitId());

    Int_t sh_region = id.region();
    // Int_t sh_ring = id.ring();
    Int_t sh_roll = id.roll();
    Int_t sh_station = id.station();
    Int_t sh_layer = id.layer();
    Int_t sh_chamber = id.chamber();

    if (GEMGeometry_->idToDet(hits->detUnitId()) == nullptr) {
      std::cout << "simHit did not matched with GEMGeometry." << std::endl;
      continue;
    }

    if (!(abs(hits->particleType()) == 13))
      continue;

    // const LocalPoint p0(0., 0., 0.);
    // const GlobalPoint
    // Gp0(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(p0));
    const LocalPoint hitLP(hits->localPosition());

    const LocalPoint hitEP(hits->entryPoint());
    Int_t sh_strip = GEMGeometry_->etaPartition(hits->detUnitId())->strip(hitEP);

    // const GlobalPoint
    // hitGP(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
    // Float_t sh_l_r = hitLP.perp();
    Float_t sh_l_x = hitLP.x();
    Float_t sh_l_y = hitLP.y();
    // Float_t sh_l_z = hitLP.z();

    for (GEMRecHitCollection::const_iterator recHit = gemRecHits->begin(); recHit != gemRecHits->end(); ++recHit) {
      Float_t rh_l_x = recHit->localPosition().x();
      Float_t rh_l_xErr = recHit->localPositionError().xx();
      Float_t rh_l_y = recHit->localPosition().y();
      Float_t rh_l_yErr = recHit->localPositionError().yy();
      // Int_t  detId = (Short_t) (*recHit).gemId();
      // Int_t  bx = recHit->BunchX();
      Int_t clusterSize = recHit->clusterSize();
      Int_t firstClusterStrip = recHit->firstClusterStrip();

      GEMDetId id((*recHit).gemId());

      Short_t rh_region = (Short_t)id.region();
      // Int_t rh_ring = (Short_t) id.ring();
      Short_t rh_station = (Short_t)id.station();
      Short_t rh_layer = (Short_t)id.layer();
      Short_t rh_chamber = (Short_t)id.chamber();
      Short_t rh_roll = (Short_t)id.roll();

      LocalPoint recHitLP = recHit->localPosition();
      if (GEMGeometry_->idToDet((*recHit).gemId()) == nullptr) {
        std::cout << "This gem recHit did not matched with GEMGeometry." << std::endl;
        continue;
      }
      GlobalPoint recHitGP = GEMGeometry_->idToDet((*recHit).gemId())->surface().toGlobal(recHitLP);

      Float_t rh_g_R = recHitGP.perp();
      // Float_t rh_g_Eta = recHitGP.eta();
      // Float_t rh_g_Phi = recHitGP.phi();
      Float_t rh_g_X = recHitGP.x();
      Float_t rh_g_Y = recHitGP.y();
      Float_t rh_g_Z = recHitGP.z();
      Float_t rh_pullX = (Float_t)(rh_l_x - sh_l_x) / (rh_l_xErr);
      Float_t rh_pullY = (Float_t)(rh_l_y - sh_l_y) / (rh_l_yErr);

      std::vector<int> stripsFired;
      for (int i = firstClusterStrip; i < (firstClusterStrip + clusterSize); i++) {
        stripsFired.push_back(i);
      }

      const bool cond1(sh_region == rh_region and sh_layer == rh_layer and sh_station == rh_station);
      const bool cond2(sh_chamber == rh_chamber and sh_roll == rh_roll);
      const bool cond3(std::find(stripsFired.begin(), stripsFired.end(), (sh_strip + 1)) != stripsFired.end());

      if (cond1 and cond2 and cond3) {
        LogDebug("GEMRecHitsValidation") << " Region : " << rh_region << "\t Station : " << rh_station
                                         << "\t Layer : " << rh_layer << "\n Radius: " << rh_g_R << "\t X : " << rh_g_X
                                         << "\t Y : " << rh_g_Y << "\t Z : " << rh_g_Z << std::endl;

        int region_num = 0;
        if (rh_region == -1)
          region_num = 0;
        else if (rh_region == 1)
          region_num = 1;
        int layer_num = rh_layer - 1;
        int binX = (rh_chamber - 1) * 2 + layer_num;
        int binY = rh_roll;
        int station_num = rh_station - 1;

        // Fill normal plots.
        TString histname_suffix = TString::Format("_r%d", rh_region);
        TString simple_zr_histname = TString::Format("rh_simple_zr%s", histname_suffix.Data());
        LogDebug("GEMRecHitsValidation") << " simpleZR!\n";
        recHits_simple_zr[simple_zr_histname.Hash()]->Fill(fabs(rh_g_Z), rh_g_R);

        histname_suffix = TString::Format("_r%d_st%d", rh_region, rh_station);
        TString dcEta_histname = TString::Format("rh_dcEta%s", histname_suffix.Data());
        LogDebug("GEMRecHitsValidation") << " dcEta\n";
        recHits_dcEta[dcEta_histname.Hash()]->Fill(binX, binY);

        gem_cls_tot->Fill(clusterSize);
        gem_region_pullX[region_num]->Fill(rh_pullX);
        gem_region_pullY[region_num]->Fill(rh_pullY);
        LogDebug("GEMRecHitsValidation") << " Begin detailPlot!\n";

        if (detailPlot_) {
          gem_cls[region_num][station_num][layer_num]->Fill(clusterSize);
          gem_pullX[region_num][station_num][layer_num]->Fill(rh_pullX);
          gem_pullY[region_num][station_num][layer_num]->Fill(rh_pullY);
          gem_rh_zr[region_num][station_num][layer_num]->Fill(rh_g_Z, rh_g_R);
          gem_rh_xy[region_num][station_num][layer_num]->Fill(rh_g_X, rh_g_Y);
        }
      }
    }  // End loop on RecHits
  }    // End loop on SimHits
}
