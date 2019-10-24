#include "Validation/MuonGEMDigis/interface/GEMPadDigiClusterValidation.h"
#include <TMath.h>

GEMPadDigiClusterValidation::GEMPadDigiClusterValidation(const edm::ParameterSet &cfg) : GEMBaseValidation(cfg) {
  InputTagToken_ = consumes<GEMPadDigiClusterCollection>(cfg.getParameter<edm::InputTag>("ClusterLabel"));
  detailPlot_ = cfg.getParameter<bool>("detailPlot");
}
void GEMPadDigiClusterValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                                 edm::Run const &Run,
                                                 edm::EventSetup const &iSetup) {
  const GEMGeometry *GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  LogDebug("GEMPadDigiClusterValidation") << "Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("MuonGEMDigisV/GEMDigisTask");
  LogDebug("GEMPadDigiClusterValidation") << "ibooker set current folder\n";

  if (GEMGeometry_ == nullptr)
    return;
  int npadsGE11 =
      GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  int npadsGE21 = 0;
  int nPads = 0;

  if (GEMGeometry_->regions()[0]->stations().size() > 1 &&
      !GEMGeometry_->regions()[0]->stations()[1]->superChambers().empty()) {
    npadsGE21 =
        GEMGeometry_->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  }

  for (auto &region : GEMGeometry_->regions()) {
    int re = region->region();
    TString title_suffix = getSuffixTitle(re);
    TString histname_suffix = getSuffixName(re);
    TString simpleZR_title = TString::Format("ZR Occupancy%s; |Z|(cm) ; R(cm)", title_suffix.Data());
    TString simpleZR_histname = TString::Format("cluster_simple_zr%s", histname_suffix.Data());

    auto *simpleZR = getSimpleZR(ibooker, simpleZR_title, simpleZR_histname);
    if (simpleZR != nullptr) {
      theCluster_simple_zr[simpleZR_histname.Hash()] = simpleZR;
    }
    for (auto &station : region->stations()) {
      int st = station->station();
      TString title_suffix2 = getSuffixTitle(re, st);
      TString histname_suffix2 = getSuffixName(re, st);

      TString dcEta_title =
          TString::Format("Occupancy for detector component %s;;#eta-partition", title_suffix2.Data());
      TString dcEta_histname = TString::Format("cluster_dcEta%s", histname_suffix2.Data());

      auto *dcEta = getDCEta(ibooker, station, dcEta_title, dcEta_histname);
      if (dcEta != nullptr) {
        theCluster_dcEta[dcEta_histname.Hash()] = dcEta;
      }
    }
  }

  if (detailPlot_) {
    for (auto &region : GEMGeometry_->regions()) {
      int re = region->region();
      int region_num = (re + 1) / 2;
      for (auto &station : region->stations()) {
        int st = station->station();
        int station_num = st - 1;
        if (station_num == 0)
          nPads = npadsGE11;
        else
          nPads = npadsGE21;
        for (int la = 1; la <= 2; la++) {
          int layer_num = la - 1;
          std::string name_prefix = getSuffixName(re, st, la);
          std::string label_prefix = getSuffixTitle(re, st, la);
          theGEMCluster_phipad[region_num][station_num][layer_num] =
              ibooker.book2D(("cluster_dg_phipad" + name_prefix).c_str(),
                             ("Digi occupancy: " + label_prefix + "; phi [rad]; Pad number").c_str(),
                             280,
                             -TMath::Pi(),
                             TMath::Pi(),
                             nPads / 2,
                             0,
                             nPads);
          theGEMCluster[region_num][station_num][layer_num] =
              ibooker.book1D(("cluster_dg" + name_prefix).c_str(),
                             ("Digi occupancy per pad number: " + label_prefix + ";Pad number; entries").c_str(),
                             nPads,
                             0.5,
                             nPads + 0.5);
          theGEMCluster_bx[region_num][station_num][layer_num] =
              ibooker.book1D(("cluster_dg_bx" + name_prefix).c_str(),
                             ("Bunch crossing: " + label_prefix + "; bunch crossing ; entries").c_str(),
                             11,
                             -5.5,
                             5.5);
          theGEMCluster_zr[region_num][station_num][layer_num] =
              BookHistZR(ibooker, "cluster_dg", "Pad Digi", region_num, station_num, layer_num);
          theGEMCluster_xy[region_num][station_num][layer_num] =
              BookHistXY(ibooker, "cluster_dg", "Pad Digi", region_num, station_num, layer_num);
          TString xy_name = TString::Format("cluster_dg_xy%s_odd", name_prefix.c_str());
          TString xy_title = TString::Format("Digi XY occupancy %s at odd chambers", label_prefix.c_str());
          theGEMCluster_xy_ch[xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360, 360, 360, -360, 360);
          xy_name = TString::Format("cluster_dg_xy%s_even", name_prefix.c_str());
          xy_title = TString::Format("Digi XY occupancy %s at even chambers", label_prefix.c_str());
          theGEMCluster_xy_ch[xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360, 360, 360, -360, 360);
        }
      }
    }
  }
}

GEMPadDigiClusterValidation::~GEMPadDigiClusterValidation() {}

void GEMPadDigiClusterValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const GEMGeometry *GEMGeometry_;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry> &e) {
    edm::LogError("GEMPadDigiClusterValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }
  edm::Handle<GEMPadDigiClusterCollection> gem_digis;
  e.getByToken(InputTagToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMPadDigiClusterValidation") << "Cannot get pads by label GEMPadToken.";
  }

  for (GEMPadDigiClusterCollection::DigiRangeIterator cItr = gem_digis->begin(); cItr != gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;

    const GeomDet *gdet = GEMGeometry_->idToDet(id);
    if (gdet == nullptr) {
      std::cout << "Getting DetId failed. Discard this gem pad hit.Maybe it "
                   "comes from unmatched geometry."
                << std::endl;
      continue;
    }
    const BoundPlane &surface = gdet->surface();
    const GEMEtaPartition *roll = GEMGeometry_->etaPartition(id);

    int re = id.region();
    int la = id.layer();
    int st = id.station();
    Short_t chamber = (Short_t)id.chamber();
    Short_t nroll = (Short_t)id.roll();
    GEMPadDigiClusterCollection::const_iterator digiItr;

    // loop over digis of given roll
    //
    for (digiItr = (*cItr).second.first; digiItr != (*cItr).second.second; ++digiItr) {
      Short_t pad = (Short_t)digiItr->pads()[0];
      Short_t bx = (Short_t)digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pads()[0]);

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t)gp.perp();
      Float_t g_phi = (Float_t)gp.phi();
      Float_t g_x = (Float_t)gp.x();
      Float_t g_y = (Float_t)gp.y();
      Float_t g_z = (Float_t)gp.z();
      edm::LogInfo("GEMPadDIGIValidation") << "Global x " << g_x << "Global y " << g_y << "\n";
      edm::LogInfo("GEMPadDIGIValidation") << "Global pad " << pad << "Global phi " << g_phi << std::endl;
      edm::LogInfo("GEMPadDIGIValidation") << "Global bx " << bx << std::endl;

      int region_num = (re + 1) / 2;
      int station_num = st - 1;
      int layer_num = la - 1;
      int binX = (chamber - 1) * 2 + layer_num;
      int binY = nroll;

      // Fill normal plots.
      TString histname_suffix = getSuffixName(re);
      TString simple_zr_histname = TString::Format("cluster_simple_zr%s", histname_suffix.Data());
      theCluster_simple_zr[simple_zr_histname.Hash()]->Fill(fabs(g_z), g_r);

      histname_suffix = getSuffixName(re, st);
      TString dcEta_histname = TString::Format("cluster_dcEta%s", histname_suffix.Data());
      theCluster_dcEta[dcEta_histname.Hash()]->Fill(binX, binY);

      if (detailPlot_) {
        theGEMCluster_xy[region_num][station_num][layer_num]->Fill(g_x, g_y);
        theGEMCluster_phipad[region_num][station_num][layer_num]->Fill(g_phi, pad);
        theGEMCluster[region_num][station_num][layer_num]->Fill(pad);
        theGEMCluster_bx[region_num][station_num][layer_num]->Fill(bx);
        theGEMCluster_zr[region_num][station_num][layer_num]->Fill(g_z, g_r);
        std::string name_prefix = getSuffixName(re, st, la);
        TString hname;
        if (chamber % 2 == 0) {
          hname = TString::Format("cluster_dg_xy%s_even", name_prefix.c_str());
        } else {
          hname = TString::Format("cluster_dg_xy%s_odd", name_prefix.c_str());
        }
        theGEMCluster_xy_ch[hname.Hash()]->Fill(g_x, g_y);
      }
    }
  }
}
