#include "Validation/MuonME0Validation/interface/ME0RecHitsValidation.h"
#include <TMath.h>

ME0RecHitsValidation::ME0RecHitsValidation(const edm::ParameterSet &cfg) : ME0BaseValidation(cfg) {
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_RecHit = consumes<ME0RecHitCollection>(cfg.getParameter<edm::InputTag>("recHitInputLabel"));
}

void ME0RecHitsValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                          edm::Run const &Run,
                                          edm::EventSetup const &iSetup) {
  LogDebug("MuonME0RecHitsValidation") << "Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0RecHitsV/ME0RecHitsTask");

  unsigned int nregion = 2;

  edm::LogInfo("MuonME0RecHitsValidation") << "+++ Info : # of region : " << nregion << std::endl;

  LogDebug("MuonME0RecHitsValidation") << "+++ Info : finish to get geometry information from ES.\n";

  for (unsigned int region_num = 0; region_num < nregion; region_num++) {
    me0_rh_zr[region_num] = BookHistZR(ibooker, "me0_rh_tot", "Digi", region_num);
    for (unsigned int layer_num = 0; layer_num < 6; layer_num++) {
      me0_rh_xy[region_num][layer_num] = BookHistXY(ibooker, "me0_rh", "RecHit", region_num, layer_num);

      std::string histo_name_DeltaX =
          std::string("me0_rh_DeltaX_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_name_DeltaY =
          std::string("me0_rh_DeltaY_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_label_DeltaX = "RecHit Delta X : region" + regionLabel[region_num] + " layer " +
                                       layerLabel[layer_num] + " " + " ; x_{SimHit} - x_{RecHit} ; entries";
      std::string histo_label_DeltaY = "RecHit Delta Y : region" + regionLabel[region_num] + " layer " +
                                       layerLabel[layer_num] + " " + " ; y_{SimHit} - y_{RecHit} ; entries";

      me0_rh_DeltaX[region_num][layer_num] =
          ibooker.book1D(histo_name_DeltaX.c_str(), histo_label_DeltaX.c_str(), 100, -10, 10);
      me0_rh_DeltaY[region_num][layer_num] =
          ibooker.book1D(histo_name_DeltaY.c_str(), histo_label_DeltaY.c_str(), 100, -10, 10);

      std::string histo_name_PullX =
          std::string("me0_rh_PullX_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_name_PullY =
          std::string("me0_rh_PullY_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_label_PullX = "RecHit Pull X : region" + regionLabel[region_num] + " layer " +
                                      layerLabel[layer_num] + " " +
                                      " ; #frac{x_{SimHit} - x_{RecHit}}{#sigma_{x,RecHit}} ; entries";
      std::string histo_label_PullY = "RecHit Pull Y : region" + regionLabel[region_num] + " layer " +
                                      layerLabel[layer_num] + " " +
                                      " ; #frac{y_{SimHit} - y_{RecHit}}{#sigma_{y,RecHit}} ; entries";

      me0_rh_PullX[region_num][layer_num] =
          ibooker.book1D(histo_name_PullX.c_str(), histo_label_DeltaX.c_str(), 100, -10, 10);
      me0_rh_PullY[region_num][layer_num] =
          ibooker.book1D(histo_name_PullY.c_str(), histo_label_DeltaY.c_str(), 100, -10, 10);
    }
  }
}

ME0RecHitsValidation::~ME0RecHitsValidation() {}

void ME0RecHitsValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const ME0Geometry *ME0Geometry_ = &iSetup.getData(geomToken_);

  edm::Handle<edm::PSimHitContainer> ME0Hits;
  e.getByToken(InputTagToken_, ME0Hits);

  edm::Handle<ME0RecHitCollection> ME0RecHits;
  e.getByToken(InputTagToken_RecHit, ME0RecHits);

  if (!ME0Hits.isValid() || !ME0RecHits.isValid()) {
    edm::LogError("ME0RecHitsValidation") << "Cannot get ME0Hits/ME0RecHits by Token simInputTagToken";
    return;
  }

  for (auto hits = ME0Hits->begin(); hits != ME0Hits->end(); hits++) {
    const ME0DetId id(hits->detUnitId());
    Int_t sh_region = id.region();
    Int_t sh_layer = id.layer();
    Int_t sh_station = 0;
    Int_t sh_chamber = id.chamber();
    Int_t sh_roll = id.roll();

    // Int_t even_odd = id.chamber()%2;
    if (ME0Geometry_->idToDet(hits->detUnitId()) == nullptr) {
      edm::LogInfo("ME0RecHitsValidation") << "simHit did not matched with GEMGeometry." << std::endl;
      continue;
    }

    const LocalPoint hitLP(hits->localPosition());

    for (ME0RecHitCollection::const_iterator recHit = ME0RecHits->begin(); recHit != ME0RecHits->end(); ++recHit) {
      Float_t x = recHit->localPosition().x();
      Float_t xErr = recHit->localPositionError().xx();
      Float_t y = recHit->localPosition().y();
      Float_t yErr = recHit->localPositionError().yy();
      // Float_t detId = (Short_t) (*recHit).me0Id();
      Float_t bx = 0;
      Float_t clusterSize = 0;
      Float_t firstClusterStrip = 0;

      ME0DetId id((*recHit).me0Id());

      Short_t region = (Short_t)id.region();
      Short_t station = 0;
      Short_t layer = (Short_t)id.layer();
      Short_t chamber = (Short_t)id.chamber();
      Short_t roll = (Short_t)id.roll();

      LocalPoint rhLP = recHit->localPosition();
      GlobalPoint rhGP = ME0Geometry_->idToDet((*recHit).me0Id())->surface().toGlobal(rhLP);

      Float_t globalR = rhGP.perp();
      Float_t globalX = rhGP.x();
      Float_t globalY = rhGP.y();
      Float_t globalZ = rhGP.z();

      Float_t x_sim = hitLP.x();
      Float_t y_sim = hitLP.y();
      Float_t pullX = (x_sim - x) / sqrt(xErr);
      Float_t pullY = (y_sim - y) / sqrt(yErr);

      if (bx != 0)
        continue;

      std::vector<int> stripsFired;
      for (int i = firstClusterStrip; i < (firstClusterStrip + clusterSize); i++) {
        stripsFired.push_back(i);
      }

      const bool cond1(sh_region == region and sh_layer == layer and sh_station == station);
      const bool cond2(sh_chamber == chamber and sh_roll == roll);
      const bool cond3(
          (std::find(stripsFired.begin(), stripsFired.end(), (firstClusterStrip + 1)) != stripsFired.end()) or
          clusterSize == 0);

      int region_num = 0;
      if (region == -1)
        region_num = 0;
      else if (region == 1)
        region_num = 1;
      int layer_num = layer - 1;

      if (cond1 and cond2 and cond3) {
        me0_rh_xy[region_num][layer_num]->Fill(globalX, globalY);
        me0_rh_zr[region_num]->Fill(globalZ, globalR);

        me0_rh_DeltaX[region_num][layer_num]->Fill(x_sim - x);
        me0_rh_DeltaY[region_num][layer_num]->Fill(y_sim - y);
        me0_rh_PullX[region_num][layer_num]->Fill(pullX);
        me0_rh_PullY[region_num][layer_num]->Fill(pullY);
      }
    }
  }
}
