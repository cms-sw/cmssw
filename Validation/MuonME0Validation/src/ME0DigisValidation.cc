#include "Validation/MuonME0Validation/interface/ME0DigisValidation.h"
#include <TMath.h>

ME0DigisValidation::ME0DigisValidation(const edm::ParameterSet &cfg) : ME0BaseValidation(cfg) {
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_Digi = consumes<ME0DigiPreRecoCollection>(cfg.getParameter<edm::InputTag>("digiInputLabel"));
  sigma_x_ = cfg.getParameter<double>("sigma_x");
  sigma_y_ = cfg.getParameter<double>("sigma_y");
}

void ME0DigisValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                        edm::Run const &Run,
                                        edm::EventSetup const &iSetup) {
  LogDebug("MuonME0DigisValidation") << "Info: Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0DigisV/ME0DigisTask");

  unsigned int nregion = 2;

  edm::LogInfo("MuonME0DigisValidation") << "+++ Info : # of region: " << nregion << std::endl;

  LogDebug("MuonME0DigisValidation") << "+++ Info : finish to get geometry information from ES.\n";

  num_evts = ibooker.book1D("num_evts", "Number of events; ; Number of events", 1, 0, 2);

  me0_strip_dg_x_local_tot =
      ibooker.book1D("me0_strip_dg_x_local_tot", "Local X; X_{local} [cm]; Entries", 60, -30.0, +30.0);
  me0_strip_dg_y_local_tot =
      ibooker.book1D("me0_strip_dg_y_local_tot", "Local Y; Y_{local} [cm]; Entries", 100, -50.0, +50.0);
  me0_strip_dg_time_tot = ibooker.book1D("me0_strip_dg_time_tot", "ToF; ToF [ns]; Entries", 400, -200, +200);

  me0_strip_dg_dx_local_tot_Muon =
      ibooker.book1D("me0_strip_dg_dx_local_tot", "Local DeltaX; #Delta X_{local} [cm]; Entries", 50, -0.1, +0.1);
  me0_strip_dg_dy_local_tot_Muon =
      ibooker.book1D("me0_strip_dg_dy_local_tot", "Local DeltaY; #Delta Y_{local} [cm]; Entries", 500, -10.0, +10.0);
  me0_strip_dg_dphi_global_tot_Muon = ibooker.book1D(
      "me0_strip_dg_dphi_global_tot", "Global DeltaPhi; #Delta #phi_{global} [rad]; Entries", 50, -0.01, +0.01);
  me0_strip_dg_dtime_tot_Muon =
      ibooker.book1D("me0_strip_dg_dtime_tot", "DeltaToF; #Delta ToF [ns]; Entries", 50, -5, +5);

  me0_strip_dg_dphi_vs_phi_global_tot_Muon = ibooker.book2D("me0_strip_dg_dphi_vs_phi_global_tot",
                                                            "Global DeltaPhi vs. Phi; #phi_{global} [rad]; #Delta "
                                                            "#phi_{global} [rad]",
                                                            72,
                                                            -M_PI,
                                                            +M_PI,
                                                            50,
                                                            -0.01,
                                                            +0.01);

  me0_strip_dg_den_eta_tot = ibooker.book1D("me0_strip_dg_den_eta_tot", "Denominator; #eta; Entries", 12, 1.8, 3.0);
  me0_strip_dg_num_eta_tot = ibooker.book1D("me0_strip_dg_num_eta_tot", "Numerator; #eta; Entries", 12, 1.8, 3.0);

  float bins[] = {62.3, 70.0, 77.7, 87.1, 96.4, 108.2, 119.9, 134.7, 149.5};
  int binnum = sizeof(bins) / sizeof(float) - 1;

  me0_strip_dg_bkg_rad_tot =
      ibooker.book1D("me0_strip_dg_bkg_radius_tot", "Total neutron background; Radius [cm]; Entries", binnum, bins);
  me0_strip_dg_bkgElePos_rad = ibooker.book1D(
      "me0_strip_dg_bkgElePos_radius", "Neutron background: electrons+positrons; Radius [cm]; Entries", binnum, bins);
  me0_strip_dg_bkgNeutral_rad = ibooker.book1D(
      "me0_strip_dg_bkgNeutral_radius", "Neutron background: gammas+neutrons; Radius [cm]; Entries", binnum, bins);

  me0_strip_exp_bkg_rad_tot = ibooker.book1D("me0_strip_exp_bkg_radius_tot",
                                             "Total expected neutron background; Radius [cm]; Hit Rate [Hz/cm^{2}]",
                                             binnum,
                                             bins);
  me0_strip_exp_bkgElePos_rad = ibooker.book1D("me0_strip_exp_bkgElePos_radius",
                                               "Expected neutron background: electrons+positrons; Radius "
                                               "[cm]; Hit Rate [Hz/cm^{2}]",
                                               binnum,
                                               bins);
  me0_strip_exp_bkgNeutral_rad = ibooker.book1D("me0_strip_exp_bkgNeutral_radius",
                                                "Expected neutron background: gammas+neutrons; Radius "
                                                "[cm]; Hit Rate [Hz/cm^{2}]",
                                                binnum,
                                                bins);

  std::vector<double> neuBkg, eleBkg;
  neuBkg.push_back(899644.0);
  neuBkg.push_back(-30841.0);
  neuBkg.push_back(441.28);
  neuBkg.push_back(-3.3405);
  neuBkg.push_back(0.0140588);
  neuBkg.push_back(-3.11473e-05);
  neuBkg.push_back(2.83736e-08);
  eleBkg.push_back(4.68590e+05);
  eleBkg.push_back(-1.63834e+04);
  eleBkg.push_back(2.35700e+02);
  eleBkg.push_back(-1.77706e+00);
  eleBkg.push_back(7.39960e-03);
  eleBkg.push_back(-1.61448e-05);
  eleBkg.push_back(1.44368e-08);

  for (int i = 1; i < me0_strip_exp_bkgNeutral_rad->getTH1F()->GetSize(); i++) {
    double pos = me0_strip_exp_bkgNeutral_rad->getTH1F()->GetBinCenter(i);
    double neutral = 0;
    double charged = 0;

    double pos_helper = 1.0;
    for (int j = 0; j < 7; ++j) {
      neutral += neuBkg[j] * pos_helper;
      charged += eleBkg[j] * pos_helper;
      pos_helper *= pos;
    }

    me0_strip_exp_bkgNeutral_rad->setBinContent(i, neutral);
    me0_strip_exp_bkgElePos_rad->setBinContent(i, charged);
    me0_strip_exp_bkg_rad_tot->setBinContent(i, neutral + charged);
  }

  for (unsigned int region_num = 0; region_num < nregion; region_num++) {
    me0_strip_dg_zr_tot[region_num] = BookHistZR(ibooker, "me0_strip_dg_tot", "Digi", region_num);
    me0_strip_dg_zr_tot_Muon[region_num] = BookHistZR(ibooker, "me0_strip_dg_tot_Muon", "Digi Muon", region_num);
    for (unsigned int layer_num = 0; layer_num < 6; layer_num++) {
      std::string hist_name_for_dx_local =
          std::string("me0_strip_dg_dx_local") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_name_for_dy_local =
          std::string("me0_strip_dg_dy_local") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_name_for_dphi_global =
          std::string("me0_strip_dg_dphi_global") + regionLabel[region_num] + "_l" + layerLabel[layer_num];

      std::string hist_name_for_den_eta =
          std::string("me0_strip_dg_den_eta") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string hist_name_for_num_eta =
          std::string("me0_strip_dg_num_eta") + regionLabel[region_num] + "_l" + layerLabel[layer_num];

      std::string hist_label_for_dx_local = "Local DeltaX: region" + regionLabel[region_num] + " layer " +
                                            layerLabel[layer_num] + " ; #Delta X_{local} [cm]; Entries";
      std::string hist_label_for_dy_local = "Local DeltaY: region" + regionLabel[region_num] + " layer " +
                                            layerLabel[layer_num] + " ; #Delta Y_{local} [cm]; Entries";
      std::string hist_label_for_dphi_global = "Global DeltaPhi: region" + regionLabel[region_num] + " layer " +
                                               layerLabel[layer_num] + " ; #Delta #phi_{global} [rad]; Entries";

      std::string hist_label_for_den_eta =
          "Denominator: region" + regionLabel[region_num] + " layer " + layerLabel[layer_num] + " ; #eta; Entries";
      std::string hist_label_for_num_eta =
          "Numerator: region" + regionLabel[region_num] + " layer " + layerLabel[layer_num] + " ; #eta; Entries";

      me0_strip_dg_xy[region_num][layer_num] = BookHistXY(ibooker, "me0_strip_dg", "Digi", region_num, layer_num);
      me0_strip_dg_xy_Muon[region_num][layer_num] =
          BookHistXY(ibooker, "me0_strip_dg_Muon", "Digi Muon", region_num, layer_num);

      me0_strip_dg_dx_local_Muon[region_num][layer_num] =
          ibooker.book1D(hist_name_for_dx_local.c_str(), hist_label_for_dx_local.c_str(), 50, -0.1, +0.1);
      me0_strip_dg_dy_local_Muon[region_num][layer_num] =
          ibooker.book1D(hist_name_for_dy_local.c_str(), hist_label_for_dy_local.c_str(), 500, -10.0, +10.0);
      me0_strip_dg_dphi_global_Muon[region_num][layer_num] =
          ibooker.book1D(hist_name_for_dphi_global.c_str(), hist_label_for_dphi_global.c_str(), 50, -0.01, +0.01);

      me0_strip_dg_den_eta[region_num][layer_num] =
          ibooker.book1D(hist_name_for_den_eta, hist_label_for_den_eta, 12, 1.8, 3.0);
      me0_strip_dg_num_eta[region_num][layer_num] =
          ibooker.book1D(hist_name_for_num_eta, hist_label_for_num_eta, 12, 1.8, 3.0);
    }
  }
}

ME0DigisValidation::~ME0DigisValidation() {}

void ME0DigisValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const ME0Geometry *ME0Geometry_ = &iSetup.getData(geomToken_);

  edm::Handle<edm::PSimHitContainer> ME0Hits;
  e.getByToken(InputTagToken_, ME0Hits);

  edm::Handle<ME0DigiPreRecoCollection> ME0Digis;
  e.getByToken(InputTagToken_Digi, ME0Digis);

  if (!ME0Hits.isValid() || !ME0Digis.isValid()) {
    edm::LogError("ME0DigisValidation") << "Cannot get ME0Hits/ME0Digis by Token simInputTagToken";
    return;
  }

  num_evts->Fill(1);
  bool toBeCounted1 = true;

  for (ME0DigiPreRecoCollection::DigiRangeIterator cItr = ME0Digis->begin(); cItr != ME0Digis->end(); cItr++) {
    ME0DetId id = (*cItr).first;

    const GeomDet *gdet = ME0Geometry_->idToDet(id);
    if (gdet == nullptr) {
      edm::LogWarning("ME0DigisValidation") << "Getting DetId failed. Discard this gem strip hit. Maybe it comes "
                                               "from unmatched geometry.";
      continue;
    }
    const BoundPlane &surface = gdet->surface();

    int region = (int)id.region();
    int layer = (int)id.layer();
    int chamber = (int)id.chamber();
    int roll = (int)id.roll();

    ME0DigiPreRecoCollection::const_iterator digiItr;
    for (digiItr = (*cItr).second.first; digiItr != (*cItr).second.second; ++digiItr) {
      LocalPoint lp(digiItr->x(), digiItr->y(), 0);

      GlobalPoint gp = surface.toGlobal(lp);

      float g_r = (float)gp.perp();
      float g_x = (float)gp.x();
      float g_y = (float)gp.y();
      float g_z = (float)gp.z();

      int particleType = digiItr->pdgid();
      int isPrompt = digiItr->prompt();

      float timeOfFlight = digiItr->tof();

      me0_strip_dg_x_local_tot->Fill(lp.x());
      me0_strip_dg_y_local_tot->Fill(lp.y());
      me0_strip_dg_time_tot->Fill(timeOfFlight);

      // fill hist
      int region_num = 0;
      if (region == -1)
        region_num = 0;
      else if (region == 1)
        region_num = 1;
      int layer_num = layer - 1;

      bool toBeCounted2 = true;

      if (isPrompt == 1 && abs(particleType) == 13) {
        me0_strip_dg_zr_tot_Muon[region_num]->Fill(g_z, g_r);
        me0_strip_dg_xy_Muon[region_num][layer_num]->Fill(g_x, g_y);

        for (auto hits = ME0Hits->begin(); hits != ME0Hits->end(); hits++) {
          int particleType_sh = hits->particleType();
          int evtId_sh = hits->eventId().event();
          int bx_sh = hits->eventId().bunchCrossing();
          int procType_sh = hits->processType();

          if (!(abs(particleType_sh) == 13 && evtId_sh == 0 && bx_sh == 0 && procType_sh == 0))
            continue;

          const ME0DetId id(hits->detUnitId());
          int region_sh = id.region();
          int layer_sh = id.layer();
          int chamber_sh = id.chamber();
          int roll_sh = id.roll();

          int region_sh_num = 0;
          if (region_sh == -1)
            region_sh_num = 0;
          else if (region_sh == 1)
            region_sh_num = 1;
          int layer_sh_num = layer_sh - 1;

          LocalPoint lp_sh = hits->localPosition();
          GlobalPoint gp_sh = ME0Geometry_->idToDet(id)->surface().toGlobal(lp_sh);

          if (toBeCounted1) {
            me0_strip_dg_den_eta[region_sh_num][layer_sh_num]->Fill(fabs(gp_sh.eta()));
            me0_strip_dg_den_eta_tot->Fill(fabs(gp_sh.eta()));
          }

          if (!(region == region_sh && layer == layer_sh && chamber == chamber_sh && roll == roll_sh))
            continue;

          float dx_loc = lp_sh.x() - lp.x();
          float dy_loc = lp_sh.y() - lp.y();
          float dphi_glob = gp_sh.phi() - gp.phi();
          float deta_glob = gp_sh.eta() - gp.eta();

          if (!(fabs(dphi_glob) < 3 * sigma_x_ && fabs(deta_glob) < 3 * sigma_y_))
            continue;

          float timeOfFlight_sh = hits->tof();
          const LocalPoint centralLP(0., 0., 0.);
          const GlobalPoint centralGP(ME0Geometry_->idToDet(id)->surface().toGlobal(centralLP));
          float centralTOF(centralGP.mag() / 29.98);  // speed of light
          float timeOfFlight_sh_corr = timeOfFlight_sh - centralTOF;

          me0_strip_dg_dx_local_Muon[region_num][layer_num]->Fill(dx_loc);
          me0_strip_dg_dy_local_Muon[region_num][layer_num]->Fill(dy_loc);
          me0_strip_dg_dphi_global_Muon[region_num][layer_num]->Fill(dphi_glob);

          me0_strip_dg_dx_local_tot_Muon->Fill(dx_loc);
          me0_strip_dg_dy_local_tot_Muon->Fill(dy_loc);
          me0_strip_dg_dphi_global_tot_Muon->Fill(dphi_glob);

          me0_strip_dg_dphi_vs_phi_global_tot_Muon->Fill(gp_sh.phi(), dphi_glob);
          me0_strip_dg_dtime_tot_Muon->Fill(timeOfFlight - timeOfFlight_sh_corr);

          if (toBeCounted2) {
            me0_strip_dg_num_eta[region_num][layer_num]->Fill(fabs(gp_sh.eta()));
            me0_strip_dg_num_eta_tot->Fill(fabs(gp_sh.eta()));
          }
          toBeCounted2 = false;

        }  // loop SH

        toBeCounted1 = false;

      } else {
        me0_strip_dg_zr_tot[region_num]->Fill(g_z, g_r);
        me0_strip_dg_xy[region_num][layer_num]->Fill(g_x, g_y);
      }

      if ((abs(particleType) == 11 || abs(particleType) == 22 || abs(particleType) == 2112) && isPrompt == 0)
        me0_strip_dg_bkg_rad_tot->Fill(fabs(gp.perp()));
      if ((abs(particleType) == 11) && isPrompt == 0)
        me0_strip_dg_bkgElePos_rad->Fill(fabs(gp.perp()));
      if ((abs(particleType) == 22 || abs(particleType) == 2112) && isPrompt == 0)
        me0_strip_dg_bkgNeutral_rad->Fill(fabs(gp.perp()));

    }  // loop DG
  }
}
