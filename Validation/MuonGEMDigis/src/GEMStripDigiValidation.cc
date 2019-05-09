#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"

#include <TMath.h>
#include <iomanip>
GEMStripDigiValidation::GEMStripDigiValidation(const edm::ParameterSet &cfg) : GEMBaseValidation(cfg) {
  InputTagToken_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("stripLabel"));
  detailPlot_ = cfg.getParameter<bool>("detailPlot");
}

void GEMStripDigiValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                            edm::Run const &Run,
                                            edm::EventSetup const &iSetup) {
  const GEMGeometry *GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  LogDebug("GEMStripDigiValidation") << "Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("MuonGEMDigisV/GEMDigisTask");
  LogDebug("GEMStripDigiValidation") << "ibooker set current folder\n";

  LogDebug("GEMStripDigiValidation") << "nregions set.\n";
  LogDebug("GEMStripDigiValidation") << "nstations set.\n";
  int nstripsGE11 = 384;
  int nstripsGE21 = 768;

  LogDebug("GEMStripDigiValidation") << "Successfully binning set.\n";

  int nstrips = 0;

  for (auto &region : GEMGeometry_->regions()) {
    int re = region->region();
    TString title_suffix = getSuffixTitle(re);
    TString histname_suffix = getSuffixName(re);
    TString simpleZR_title = TString::Format("ZR Occupancy%s; |Z|(cm) ; R(cm)", title_suffix.Data());
    TString simpleZR_histname = TString::Format("strip_simple_zr%s", histname_suffix.Data());

    auto *simpleZR = getSimpleZR(ibooker, simpleZR_title, simpleZR_histname);
    if (simpleZR != nullptr) {
      theStrip_simple_zr[simpleZR_histname.Hash()] = simpleZR;
    }

    for (auto &station : region->stations()) {
      int st = station->station();
      TString title_suffix2 = getSuffixTitle(re, st);
      TString histname_suffix2 = getSuffixName(re, st);

      TString dcEta_title =
          TString::Format("Occupancy for detector component %s;;#eta-partition", title_suffix2.Data());
      TString dcEta_histname = TString::Format("strip_dcEta%s", histname_suffix2.Data());

      auto *dcEta = getDCEta(ibooker, station, dcEta_title, dcEta_histname);
      if (dcEta != nullptr) {
        theStrip_dcEta[dcEta_histname.Hash()] = dcEta;
      }
    }
  }

  // Booking detail plot.
  if (detailPlot_) {
    for (auto &region : GEMGeometry_->regions()) {
      for (auto &station : region->stations()) {
        for (int la = 1; la <= 2; la++) {
          int re = region->region();
          int st = station->station();
          int region_num = (re + 1) / 2;
          int station_num = st - 1;
          int layer_num = la - 1;

          if (st == 1)
            nstrips = nstripsGE11;
          else
            nstrips = nstripsGE21;
          std::string name_prefix = getSuffixName(re, st, la);
          std::string label_prefix = getSuffixTitle(re, st, la);
          theStrip_phistrip[region_num][station_num][layer_num] =
              ibooker.book2D(("strip_dg_phistrip" + name_prefix).c_str(),
                             ("Digi occupancy: " + label_prefix + "; phi [rad];strip number").c_str(),
                             280,
                             -TMath::Pi(),
                             TMath::Pi(),
                             nstrips / 2,
                             0,
                             nstrips);
          theStrip[region_num][station_num][layer_num] =
              ibooker.book1D(("strip_dg" + name_prefix).c_str(),
                             ("Digi occupancy per stip number: " + label_prefix + ";strip number; entries").c_str(),
                             nstrips,
                             0.5,
                             nstrips + 0.5);
          theStrip_bx[region_num][station_num][layer_num] =
              ibooker.book1D(("strip_dg_bx" + name_prefix).c_str(),
                             ("Bunch crossing: " + label_prefix + "; bunch crossing ; entries").c_str(),
                             11,
                             -5.5,
                             5.5);
          theStrip_zr[region_num][station_num][layer_num] =
              BookHistZR(ibooker, "strip_dg", "Strip Digi", region_num, station_num, layer_num);
          theStrip_xy[region_num][station_num][layer_num] =
              BookHistXY(ibooker, "strip_dg", "Strip Digi", region_num, station_num, layer_num);
          TString xy_name = TString::Format("strip_dg_xy%s_odd", name_prefix.c_str());
          TString xy_title = TString::Format("Digi XY occupancy %s at odd chambers", label_prefix.c_str());
          theStrip_xy_ch[xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360, 360, 360, -360, 360);
          xy_name = TString::Format("strip_dg_xy%s_even", name_prefix.c_str());
          xy_title = TString::Format("Digi XY occupancy %s at even chambers", label_prefix.c_str());
          theStrip_xy_ch[xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360, 360, 360, -360, 360);
        }
      }
    }
  }
  LogDebug("GEMStripDigiValidation") << "Booking End.\n";
}

GEMStripDigiValidation::~GEMStripDigiValidation() {}

void GEMStripDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const GEMGeometry *GEMGeometry_;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry> &e) {
    edm::LogError("GEMStripDigiValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }

  edm::Handle<GEMDigiCollection> gem_digis;
  e.getByToken(this->InputTagToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMStripDigiValidation") << "Cannot get strips by Token stripToken.\n";
    return;
  }
  for (GEMDigiCollection::DigiRangeIterator cItr = gem_digis->begin(); cItr != gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;

    const GeomDet *gdet = GEMGeometry_->idToDet(id);
    if (gdet == nullptr) {
      std::cout << "Getting DetId failed. Discard this gem strip hit.Maybe it "
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

    GEMDigiCollection::const_iterator digiItr;
    for (digiItr = (*cItr).second.first; digiItr != (*cItr).second.second; ++digiItr) {
      Short_t strip = (Short_t)digiItr->strip();
      Short_t bx = (Short_t)digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t)gp.perp();
      // Float_t g_eta = (Float_t) gp.eta();
      Float_t g_phi = (Float_t)gp.phi();
      Float_t g_x = (Float_t)gp.x();
      Float_t g_y = (Float_t)gp.y();
      Float_t g_z = (Float_t)gp.z();

      int region_num = (re + 1) / 2;
      int station_num = st - 1;
      int layer_num = la - 1;

      int binX = (chamber - 1) * 2 + layer_num;
      int binY = nroll;

      // Fill normal plots.
      TString histname_suffix = getSuffixName(re);
      TString simple_zr_histname = TString::Format("strip_simple_zr%s", histname_suffix.Data());
      theStrip_simple_zr[simple_zr_histname.Hash()]->Fill(fabs(g_z), g_r);

      histname_suffix = getSuffixName(re, st);
      TString dcEta_histname = TString::Format("strip_dcEta%s", histname_suffix.Data());
      theStrip_dcEta[dcEta_histname.Hash()]->Fill(binX, binY);

      // Fill detail plots.
      if (detailPlot_) {
        if (theStrip_xy[region_num][station_num][layer_num] != nullptr) {
          theStrip_xy[region_num][station_num][layer_num]->Fill(g_x, g_y);
          theStrip_phistrip[region_num][station_num][layer_num]->Fill(g_phi, strip);
          theStrip[region_num][station_num][layer_num]->Fill(strip);
          theStrip_bx[region_num][station_num][layer_num]->Fill(bx);
          theStrip_zr[region_num][station_num][layer_num]->Fill(g_z, g_r);

          std::string name_prefix = getSuffixName(re, st, la);
          TString hname;
          if (chamber % 2 == 0) {
            hname = TString::Format("strip_dg_xy%s_even", name_prefix.c_str());
          } else {
            hname = TString::Format("strip_dg_xy%s_odd", name_prefix.c_str());
          }
          theStrip_xy_ch[hname.Hash()]->Fill(g_x, g_y);
        } else {
          std::cout << "Error is occued when histograms is called." << std::endl;
        }
      }
    }
  }
}
