#include "Validation/SiOuterTrackerV/interface/OuterTrackerMCHarvester.h"

OuterTrackerMCHarvester::OuterTrackerMCHarvester(const edm::ParameterSet &iConfig) {}

OuterTrackerMCHarvester::~OuterTrackerMCHarvester() {}

// ------------ method called once each job just after ending the event loop
// ------------
void OuterTrackerMCHarvester::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  using namespace edm;

  // Global variables
  TF1 *fit = new TF1("fit", "gaus", -0.01, 0.01);
  TF1 *fit2 = new TF1("fit2", "gaus", -0.1, 0.1);
  TF1 *fit3 = new TF1("fit3", "gaus", -1, 1);

  std::vector<double> sigma_pt1;
  std::vector<double> error_pt1;
  std::vector<double> sigma_pt2;
  std::vector<double> error_pt2;
  std::vector<double> sigma_pt3;
  std::vector<double> error_pt3;
  std::vector<double> sigma_eta;
  std::vector<double> error_eta;
  std::vector<double> sigma_phi;
  std::vector<double> error_phi;
  std::vector<double> sigma_VtxZ;
  std::vector<double> error_VtxZ;
  std::vector<double> sigma_d0;
  std::vector<double> error_d0;

  float eta_bins[] = {0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4};
  int eta_binnum = 6;

  dbe = nullptr;
  dbe = edm::Service<DQMStore>().operator->();

  if (dbe) {
    // Find all monitor elements for histograms
    MonitorElement *meN_eta = dbe->get("SiOuterTrackerV/Tracks/Efficiency/match_tp_eta");
    MonitorElement *meD_eta = dbe->get("SiOuterTrackerV/Tracks/Efficiency/tp_eta");
    MonitorElement *meN_pt = dbe->get("SiOuterTrackerV/Tracks/Efficiency/match_tp_pt");
    MonitorElement *meD_pt = dbe->get("SiOuterTrackerV/Tracks/Efficiency/tp_pt");
    MonitorElement *meN_pt_zoom = dbe->get("SiOuterTrackerV/Tracks/Efficiency/match_tp_pt_zoom");
    MonitorElement *meD_pt_zoom = dbe->get("SiOuterTrackerV/Tracks/Efficiency/tp_pt_zoom");
    MonitorElement *meN_d0 = dbe->get("SiOuterTrackerV/Tracks/Efficiency/match_tp_d0");
    MonitorElement *meD_d0 = dbe->get("SiOuterTrackerV/Tracks/Efficiency/tp_d0");
    MonitorElement *meN_VtxR = dbe->get("SiOuterTrackerV/Tracks/Efficiency/match_tp_VtxR");
    MonitorElement *meD_VtxR = dbe->get("SiOuterTrackerV/Tracks/Efficiency/tp_VtxR");
    MonitorElement *meN_VtxZ = dbe->get("SiOuterTrackerV/Tracks/Efficiency/match_tp_VtxZ");
    MonitorElement *meD_VtxZ = dbe->get("SiOuterTrackerV/Tracks/Efficiency/tp_VtxZ");

    MonitorElement *merespt_eta0to0p7_pt2to3 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta0to0p7_pt2to3");
    MonitorElement *merespt_eta0p7to1_pt2to3 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta0p7to1_pt2to3");
    MonitorElement *merespt_eta1to1p2_pt2to3 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1to1p2_pt2to3");
    MonitorElement *merespt_eta1p2to1p6_pt2to3 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1p2to1p6_pt2to3");
    MonitorElement *merespt_eta1p6to2_pt2to3 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1p6to2_pt2to3");
    MonitorElement *merespt_eta2to2p4_pt2to3 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta2to2p4_pt2to3");
    MonitorElement *merespt_eta0to0p7_pt3to8 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta0to0p7_pt3to8");
    MonitorElement *merespt_eta0p7to1_pt3to8 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta0p7to1_pt3to8");
    MonitorElement *merespt_eta1to1p2_pt3to8 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1to1p2_pt3to8");
    MonitorElement *merespt_eta1p2to1p6_pt3to8 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1p2to1p6_pt3to8");
    MonitorElement *merespt_eta1p6to2_pt3to8 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1p6to2_pt3to8");
    MonitorElement *merespt_eta2to2p4_pt3to8 = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta2to2p4_pt3to8");
    MonitorElement *merespt_eta0to0p7_pt8toInf = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta0to0p7_pt8toInf");
    MonitorElement *merespt_eta0p7to1_pt8toInf = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta0p7to1_pt8toInf");
    MonitorElement *merespt_eta1to1p2_pt8toInf = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1to1p2_pt8toInf");
    MonitorElement *merespt_eta1p2to1p6_pt8toInf =
        dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1p2to1p6_pt8toInf");
    MonitorElement *merespt_eta1p6to2_pt8toInf = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta1p6to2_pt8toInf");
    MonitorElement *merespt_eta2to2p4_pt8toInf = dbe->get("SiOuterTrackerV/Tracks/Resolution/respt_eta2to2p4_pt8toInf");

    MonitorElement *mereseta_eta0to0p7 = dbe->get("SiOuterTrackerV/Tracks/Resolution/reseta_eta0to0p7");
    MonitorElement *mereseta_eta0p7to1 = dbe->get("SiOuterTrackerV/Tracks/Resolution/reseta_eta0p7to1");
    MonitorElement *mereseta_eta1to1p2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/reseta_eta1to1p2");
    MonitorElement *mereseta_eta1p2to1p6 = dbe->get("SiOuterTrackerV/Tracks/Resolution/reseta_eta1p2to1p6");
    MonitorElement *mereseta_eta1p6to2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/reseta_eta1p6to2");
    MonitorElement *mereseta_eta2to2p4 = dbe->get("SiOuterTrackerV/Tracks/Resolution/reseta_eta2to2p4");

    MonitorElement *meresphi_eta0to0p7 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resphi_eta0to0p7");
    MonitorElement *meresphi_eta0p7to1 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resphi_eta0p7to1");
    MonitorElement *meresphi_eta1to1p2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resphi_eta1to1p2");
    MonitorElement *meresphi_eta1p2to1p6 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resphi_eta1p2to1p6");
    MonitorElement *meresphi_eta1p6to2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resphi_eta1p6to2");
    MonitorElement *meresphi_eta2to2p4 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resphi_eta2to2p4");

    MonitorElement *meresVtxZ_eta0to0p7 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resVtxZ_eta0to0p7");
    MonitorElement *meresVtxZ_eta0p7to1 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resVtxZ_eta0p7to1");
    MonitorElement *meresVtxZ_eta1to1p2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resVtxZ_eta1to1p2");
    MonitorElement *meresVtxZ_eta1p2to1p6 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resVtxZ_eta1p2to1p6");
    MonitorElement *meresVtxZ_eta1p6to2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resVtxZ_eta1p6to2");
    MonitorElement *meresVtxZ_eta2to2p4 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resVtxZ_eta2to2p4");

    MonitorElement *meresd0_eta0to0p7 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resd0_eta0to0p7");
    MonitorElement *meresd0_eta0p7to1 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resd0_eta0p7to1");
    MonitorElement *meresd0_eta1to1p2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resd0_eta1to1p2");
    MonitorElement *meresd0_eta1p2to1p6 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resd0_eta1p2to1p6");
    MonitorElement *meresd0_eta1p6to2 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resd0_eta1p6to2");
    MonitorElement *meresd0_eta2to2p4 = dbe->get("SiOuterTrackerV/Tracks/Resolution/resd0_eta2to2p4");

    if (meN_eta && meD_eta) {
      // Get the numerator and denominator histograms
      TH1F *numerator = meN_eta->getTH1F();
      TH1F *denominator = meD_eta->getTH1F();
      numerator->Sumw2();
      denominator->Sumw2();

      // Set the current directory
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalEfficiency");

      // Book the new histogram to contain the results
      MonitorElement *me_effic_eta = ibooker.book1D("EtaEfficiency",
                                                    "#eta efficiency",
                                                    numerator->GetNbinsX(),
                                                    numerator->GetXaxis()->GetXmin(),
                                                    numerator->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me_effic_eta->getTH1F()->Divide(numerator, denominator, 1., 1., "B");
      me_effic_eta->setAxisTitle("tracking particle #eta");
      me_effic_eta->getTH1F()->GetYaxis()->SetTitle("Efficiency");
      me_effic_eta->getTH1F()->SetMaximum(1.0);
      me_effic_eta->getTH1F()->SetMinimum(0.0);
      me_effic_eta->getTH1F()->SetStats(false);
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for eta efficiency cannot be found!\n";
    }

    if (meN_pt && meD_pt) {
      // Get the numerator and denominator histograms
      TH1F *numerator2 = meN_pt->getTH1F();
      numerator2->Sumw2();
      TH1F *denominator2 = meD_pt->getTH1F();
      denominator2->Sumw2();

      // Set the current directory
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalEfficiency");

      // Book the new histogram to contain the results
      MonitorElement *me_effic_pt = ibooker.book1D("PtEfficiency",
                                                   "p_{T} efficiency",
                                                   numerator2->GetNbinsX(),
                                                   numerator2->GetXaxis()->GetXmin(),
                                                   numerator2->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me_effic_pt->getTH1F()->Divide(numerator2, denominator2, 1., 1., "B");
      me_effic_pt->setAxisTitle("Tracking particle p_{T} [GeV]");
      me_effic_pt->getTH1F()->GetYaxis()->SetTitle("Efficiency");
      me_effic_pt->getTH1F()->SetMaximum(1.0);
      me_effic_pt->getTH1F()->SetMinimum(0.0);
      me_effic_pt->getTH1F()->SetStats(false);
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for pT efficiency cannot be found!\n";
    }

    if (meN_pt_zoom && meD_pt_zoom) {
      // Get the numerator and denominator histograms
      TH1F *numerator2_zoom = meN_pt_zoom->getTH1F();
      numerator2_zoom->Sumw2();
      TH1F *denominator2_zoom = meD_pt_zoom->getTH1F();
      denominator2_zoom->Sumw2();

      // Set the current directory
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalEfficiency");

      // Book the new histogram to contain the results
      MonitorElement *me_effic_pt_zoom = ibooker.book1D("PtEfficiency_zoom",
                                                        "p_{T} efficiency",
                                                        numerator2_zoom->GetNbinsX(),
                                                        numerator2_zoom->GetXaxis()->GetXmin(),
                                                        numerator2_zoom->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me_effic_pt_zoom->getTH1F()->Divide(numerator2_zoom, denominator2_zoom, 1., 1., "B");
      me_effic_pt_zoom->setAxisTitle("Tracking particle p_{T} [GeV]");
      me_effic_pt_zoom->getTH1F()->GetYaxis()->SetTitle("Efficiency");
      me_effic_pt_zoom->getTH1F()->SetMaximum(1.0);
      me_effic_pt_zoom->getTH1F()->SetMinimum(0.0);
      me_effic_pt_zoom->getTH1F()->SetStats(false);
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for zoom pT efficiency cannot be found!\n";
    }

    if (meN_d0 && meD_d0) {
      // Get the numerator and denominator histograms
      TH1F *numerator5 = meN_d0->getTH1F();
      numerator5->Sumw2();
      TH1F *denominator5 = meD_d0->getTH1F();
      denominator5->Sumw2();

      // Set the current directory
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalEfficiency");

      // Book the new histogram to contain the results
      MonitorElement *me_effic_d0 = ibooker.book1D("d0Efficiency",
                                                   "d_{0} efficiency",
                                                   numerator5->GetNbinsX(),
                                                   numerator5->GetXaxis()->GetXmin(),
                                                   numerator5->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me_effic_d0->getTH1F()->Divide(numerator5, denominator5, 1., 1., "B");
      me_effic_d0->setAxisTitle("Tracking particle d_{0} [cm]");
      me_effic_d0->getTH1F()->GetYaxis()->SetTitle("Efficiency");
      me_effic_d0->getTH1F()->SetMaximum(1.0);
      me_effic_d0->getTH1F()->SetMinimum(0.0);
      me_effic_d0->getTH1F()->SetStats(false);
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for d0 efficiency cannot be found!\n";
    }

    if (meN_VtxR && meD_VtxR) {
      // Get the numerator and denominator histograms
      TH1F *numerator6 = meN_VtxR->getTH1F();
      numerator6->Sumw2();
      TH1F *denominator6 = meD_VtxR->getTH1F();
      denominator6->Sumw2();

      // Set the current directory
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalEfficiency");

      // Book the new histogram to contain the results
      MonitorElement *me_effic_VtxR = ibooker.book1D("VtxREfficiency",
                                                     "Vtx R efficiency",
                                                     numerator6->GetNbinsX(),
                                                     numerator6->GetXaxis()->GetXmin(),
                                                     numerator6->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me_effic_VtxR->getTH1F()->Divide(numerator6, denominator6, 1., 1., "B");
      me_effic_VtxR->setAxisTitle("Tracking particle VtxR [cm]");
      me_effic_VtxR->getTH1F()->GetYaxis()->SetTitle("Efficiency");
      me_effic_VtxR->getTH1F()->SetMaximum(1.0);
      me_effic_VtxR->getTH1F()->SetMinimum(0.0);
      me_effic_VtxR->getTH1F()->SetStats(false);
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for VtxR efficiency cannot be found!\n";
    }

    if (meN_VtxZ && meD_VtxZ) {
      // Get the numerator and denominator histograms
      TH1F *numerator7 = meN_VtxZ->getTH1F();
      numerator7->Sumw2();
      TH1F *denominator7 = meD_VtxZ->getTH1F();
      denominator7->Sumw2();

      // Set the current directory
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalEfficiency");

      // Book the new histogram to contain the results
      MonitorElement *me_effic_VtxZ = ibooker.book1D("VtxZEfficiency",
                                                     "Vtx Z efficiency",
                                                     numerator7->GetNbinsX(),
                                                     numerator7->GetXaxis()->GetXmin(),
                                                     numerator7->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me_effic_VtxZ->getTH1F()->Divide(numerator7, denominator7, 1., 1., "B");
      me_effic_VtxZ->setAxisTitle("Tracking particle VtxZ [cm]");
      me_effic_VtxZ->getTH1F()->GetYaxis()->SetTitle("Efficiency");
      me_effic_VtxZ->getTH1F()->SetMaximum(1.0);
      me_effic_VtxZ->getTH1F()->SetMinimum(0.0);
      me_effic_VtxZ->getTH1F()->SetStats(false);
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for VtxZ efficiency cannot be found!\n";
    }

    if (merespt_eta0to0p7_pt2to3 && merespt_eta0p7to1_pt2to3 && merespt_eta1to1p2_pt2to3 &&
        merespt_eta1p2to1p6_pt2to3 && merespt_eta1p6to2_pt2to3 && merespt_eta2to2p4_pt2to3) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resPt1a = merespt_eta0to0p7_pt2to3->getTH1F();
      TH1F *resPt2a = merespt_eta0p7to1_pt2to3->getTH1F();
      TH1F *resPt3a = merespt_eta1to1p2_pt2to3->getTH1F();
      TH1F *resPt4a = merespt_eta1p2to1p6_pt2to3->getTH1F();
      TH1F *resPt5a = merespt_eta1p6to2_pt2to3->getTH1F();
      TH1F *resPt6a = merespt_eta2to2p4_pt2to3->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_pt1 =
          ibooker.book1D("pTResVsEta_2-3", "p_{T} resolution vs |#eta|, for p_{T}: 2-3 GeV", eta_binnum, eta_bins);
      TH1F *resPt1 = me_res_pt1->getTH1F();
      resPt1->GetXaxis()->SetTitle("tracking particle |#eta|");
      resPt1->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
      resPt1->SetMinimum(0.0);
      resPt1->SetStats(false);

      int testNumEntries1 = resPt1a->GetEntries();
      if (testNumEntries1 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resPt1a->Fit(fit2, "R");
        resPt2a->Fit(fit2, "R");
        resPt3a->Fit(fit2, "R");
        resPt4a->Fit(fit2, "R");
        resPt5a->Fit(fit2, "R");
        resPt6a->Fit(fit2, "R");
        sigma_pt1.push_back(resPt1a->GetFunction("fit2")->GetParameter(2));
        sigma_pt1.push_back(resPt2a->GetFunction("fit2")->GetParameter(2));
        sigma_pt1.push_back(resPt3a->GetFunction("fit2")->GetParameter(2));
        sigma_pt1.push_back(resPt4a->GetFunction("fit2")->GetParameter(2));
        sigma_pt1.push_back(resPt5a->GetFunction("fit2")->GetParameter(2));
        sigma_pt1.push_back(resPt6a->GetFunction("fit2")->GetParameter(2));
        error_pt1.push_back(resPt1a->GetFunction("fit2")->GetParError(2));
        error_pt1.push_back(resPt2a->GetFunction("fit2")->GetParError(2));
        error_pt1.push_back(resPt3a->GetFunction("fit2")->GetParError(2));
        error_pt1.push_back(resPt4a->GetFunction("fit2")->GetParError(2));
        error_pt1.push_back(resPt5a->GetFunction("fit2")->GetParError(2));
        error_pt1.push_back(resPt6a->GetFunction("fit2")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resPt1->SetBinContent(i + 1, sigma_pt1[i]);
          resPt1->SetBinError(i + 1, error_pt1[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for pT resolution (2-3)!\n";
        for (int i = 0; i < 6; i++) {
          resPt1->SetBinContent(i + 1, -1);
          resPt1->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution (2-3) cannot be found!\n";
    }

    if (merespt_eta0to0p7_pt3to8 && merespt_eta0p7to1_pt3to8 && merespt_eta1to1p2_pt3to8 &&
        merespt_eta1p2to1p6_pt3to8 && merespt_eta1p6to2_pt3to8 && merespt_eta2to2p4_pt3to8) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resPt1b = merespt_eta0to0p7_pt3to8->getTH1F();
      TH1F *resPt2b = merespt_eta0p7to1_pt3to8->getTH1F();
      TH1F *resPt3b = merespt_eta1to1p2_pt3to8->getTH1F();
      TH1F *resPt4b = merespt_eta1p2to1p6_pt3to8->getTH1F();
      TH1F *resPt5b = merespt_eta1p6to2_pt3to8->getTH1F();
      TH1F *resPt6b = merespt_eta2to2p4_pt3to8->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_pt2 =
          ibooker.book1D("pTResVsEta_3-8", "p_{T} resolution vs |#eta|, for p_{T}: 3-8 GeV", eta_binnum, eta_bins);
      TH1F *resPt2 = me_res_pt2->getTH1F();
      resPt2->GetXaxis()->SetTitle("tracking particle |#eta|");
      resPt2->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
      resPt2->SetMinimum(0.0);
      resPt2->SetStats(false);

      int testNumEntries2 = resPt1b->GetEntries();
      if (testNumEntries2 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resPt1b->Fit(fit2, "R");
        resPt2b->Fit(fit2, "R");
        resPt3b->Fit(fit2, "R");
        resPt4b->Fit(fit2, "R");
        resPt5b->Fit(fit2, "R");
        resPt6b->Fit(fit2, "R");
        sigma_pt2.push_back(resPt1b->GetFunction("fit2")->GetParameter(2));
        sigma_pt2.push_back(resPt2b->GetFunction("fit2")->GetParameter(2));
        sigma_pt2.push_back(resPt3b->GetFunction("fit2")->GetParameter(2));
        sigma_pt2.push_back(resPt4b->GetFunction("fit2")->GetParameter(2));
        sigma_pt2.push_back(resPt5b->GetFunction("fit2")->GetParameter(2));
        sigma_pt2.push_back(resPt6b->GetFunction("fit2")->GetParameter(2));
        error_pt2.push_back(resPt1b->GetFunction("fit2")->GetParError(2));
        error_pt2.push_back(resPt2b->GetFunction("fit2")->GetParError(2));
        error_pt2.push_back(resPt3b->GetFunction("fit2")->GetParError(2));
        error_pt2.push_back(resPt4b->GetFunction("fit2")->GetParError(2));
        error_pt2.push_back(resPt5b->GetFunction("fit2")->GetParError(2));
        error_pt2.push_back(resPt6b->GetFunction("fit2")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resPt2->SetBinContent(i + 1, sigma_pt2[i]);
          resPt2->SetBinError(i + 1, error_pt2[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for pT resolution (3-8)!\n";
        for (int i = 0; i < 6; i++) {
          resPt2->SetBinContent(i + 1, -1);
          resPt2->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution (3-8) cannot be found!\n";
    }

    if (merespt_eta0to0p7_pt8toInf && merespt_eta0p7to1_pt8toInf && merespt_eta1to1p2_pt8toInf &&
        merespt_eta1p2to1p6_pt8toInf && merespt_eta1p6to2_pt8toInf && merespt_eta2to2p4_pt8toInf) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resPt1c = merespt_eta0to0p7_pt8toInf->getTH1F();
      TH1F *resPt2c = merespt_eta0p7to1_pt8toInf->getTH1F();
      TH1F *resPt3c = merespt_eta1to1p2_pt8toInf->getTH1F();
      TH1F *resPt4c = merespt_eta1p2to1p6_pt8toInf->getTH1F();
      TH1F *resPt5c = merespt_eta1p6to2_pt8toInf->getTH1F();
      TH1F *resPt6c = merespt_eta2to2p4_pt8toInf->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_pt3 =
          ibooker.book1D("pTResVsEta_8-inf", "p_{T} resolution vs |#eta|, for p_{T}: >8 GeV", eta_binnum, eta_bins);
      TH1F *resPt3 = me_res_pt3->getTH1F();
      resPt3->GetXaxis()->SetTitle("tracking particle |#eta|");
      resPt3->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
      resPt3->SetMinimum(0.0);
      resPt3->SetStats(false);

      int testNumEntries3 = resPt1c->GetEntries();
      if (testNumEntries3 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resPt1c->Fit(fit2, "R");
        resPt2c->Fit(fit2, "R");
        resPt3c->Fit(fit2, "R");
        resPt4c->Fit(fit2, "R");
        resPt5c->Fit(fit2, "R");
        resPt6c->Fit(fit2, "R");
        sigma_pt3.push_back(resPt1c->GetFunction("fit2")->GetParameter(2));
        sigma_pt3.push_back(resPt2c->GetFunction("fit2")->GetParameter(2));
        sigma_pt3.push_back(resPt3c->GetFunction("fit2")->GetParameter(2));
        sigma_pt3.push_back(resPt4c->GetFunction("fit2")->GetParameter(2));
        sigma_pt3.push_back(resPt5c->GetFunction("fit2")->GetParameter(2));
        sigma_pt3.push_back(resPt6c->GetFunction("fit2")->GetParameter(2));
        error_pt3.push_back(resPt1c->GetFunction("fit2")->GetParError(2));
        error_pt3.push_back(resPt2c->GetFunction("fit2")->GetParError(2));
        error_pt3.push_back(resPt3c->GetFunction("fit2")->GetParError(2));
        error_pt3.push_back(resPt4c->GetFunction("fit2")->GetParError(2));
        error_pt3.push_back(resPt5c->GetFunction("fit2")->GetParError(2));
        error_pt3.push_back(resPt6c->GetFunction("fit2")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resPt3->SetBinContent(i + 1, sigma_pt3[i]);
          resPt3->SetBinError(i + 1, error_pt3[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for pT resolution (8-inf)!\n";
        for (int i = 0; i < 6; i++) {
          resPt3->SetBinContent(i + 1, -1);
          resPt3->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution (8-inf) cannot be found!\n";
    }

    if (mereseta_eta0to0p7 && mereseta_eta0p7to1 && mereseta_eta1to1p2 && mereseta_eta1p2to1p6 && mereseta_eta1p6to2 &&
        mereseta_eta2to2p4) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resEta1 = mereseta_eta0to0p7->getTH1F();
      TH1F *resEta2 = mereseta_eta0p7to1->getTH1F();
      TH1F *resEta3 = mereseta_eta1to1p2->getTH1F();
      TH1F *resEta4 = mereseta_eta1p2to1p6->getTH1F();
      TH1F *resEta5 = mereseta_eta1p6to2->getTH1F();
      TH1F *resEta6 = mereseta_eta2to2p4->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_eta = ibooker.book1D("EtaResolution", "#eta resolution vs |#eta|", eta_binnum, eta_bins);
      TH1F *resEta = me_res_eta->getTH1F();
      resEta->GetXaxis()->SetTitle("tracking particle |#eta|");
      resEta->GetYaxis()->SetTitle("#sigma(#Delta#eta)");
      resEta->SetMinimum(0.0);
      resEta->SetStats(false);

      int testNumEntries4 = resEta1->GetEntries();
      if (testNumEntries4 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resEta1->Fit(fit, "R");
        resEta2->Fit(fit, "R");
        resEta3->Fit(fit, "R");
        resEta4->Fit(fit, "R");
        resEta5->Fit(fit, "R");
        resEta6->Fit(fit, "R");
        sigma_eta.push_back(resEta1->GetFunction("fit")->GetParameter(2));
        sigma_eta.push_back(resEta2->GetFunction("fit")->GetParameter(2));
        sigma_eta.push_back(resEta3->GetFunction("fit")->GetParameter(2));
        sigma_eta.push_back(resEta4->GetFunction("fit")->GetParameter(2));
        sigma_eta.push_back(resEta5->GetFunction("fit")->GetParameter(2));
        sigma_eta.push_back(resEta6->GetFunction("fit")->GetParameter(2));
        error_eta.push_back(resEta1->GetFunction("fit")->GetParError(2));
        error_eta.push_back(resEta2->GetFunction("fit")->GetParError(2));
        error_eta.push_back(resEta3->GetFunction("fit")->GetParError(2));
        error_eta.push_back(resEta4->GetFunction("fit")->GetParError(2));
        error_eta.push_back(resEta5->GetFunction("fit")->GetParError(2));
        error_eta.push_back(resEta6->GetFunction("fit")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resEta->SetBinContent(i + 1, sigma_eta[i]);
          resEta->SetBinError(i + 1, error_eta[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for eta resolution!\n";
        for (int i = 0; i < 6; i++) {
          resEta->SetBinContent(i + 1, -1);
          resEta->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for eta resolution cannot be found!\n";
    }

    if (meresphi_eta0to0p7 && meresphi_eta0p7to1 && meresphi_eta1to1p2 && meresphi_eta1p2to1p6 && meresphi_eta1p6to2 &&
        meresphi_eta2to2p4) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resPhi1 = meresphi_eta0to0p7->getTH1F();
      TH1F *resPhi2 = meresphi_eta0p7to1->getTH1F();
      TH1F *resPhi3 = meresphi_eta1to1p2->getTH1F();
      TH1F *resPhi4 = meresphi_eta1p2to1p6->getTH1F();
      TH1F *resPhi5 = meresphi_eta1p6to2->getTH1F();
      TH1F *resPhi6 = meresphi_eta2to2p4->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_phi = ibooker.book1D("PhiResolution", "#phi resolution vs |#eta|", eta_binnum, eta_bins);
      TH1F *resPhi = me_res_phi->getTH1F();
      resPhi->GetXaxis()->SetTitle("tracking particle |#eta|");
      resPhi->GetYaxis()->SetTitle("#sigma(#Delta#phi)");
      resPhi->SetMinimum(0.0);
      resPhi->SetStats(false);

      int testNumEntries5 = resPhi1->GetEntries();
      if (testNumEntries5 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resPhi1->Fit(fit, "R");
        resPhi2->Fit(fit, "R");
        resPhi3->Fit(fit, "R");
        resPhi4->Fit(fit, "R");
        resPhi5->Fit(fit, "R");
        resPhi6->Fit(fit, "R");
        sigma_phi.push_back(resPhi1->GetFunction("fit")->GetParameter(2));
        sigma_phi.push_back(resPhi2->GetFunction("fit")->GetParameter(2));
        sigma_phi.push_back(resPhi3->GetFunction("fit")->GetParameter(2));
        sigma_phi.push_back(resPhi4->GetFunction("fit")->GetParameter(2));
        sigma_phi.push_back(resPhi5->GetFunction("fit")->GetParameter(2));
        sigma_phi.push_back(resPhi6->GetFunction("fit")->GetParameter(2));
        error_phi.push_back(resPhi1->GetFunction("fit")->GetParError(2));
        error_phi.push_back(resPhi2->GetFunction("fit")->GetParError(2));
        error_phi.push_back(resPhi3->GetFunction("fit")->GetParError(2));
        error_phi.push_back(resPhi4->GetFunction("fit")->GetParError(2));
        error_phi.push_back(resPhi5->GetFunction("fit")->GetParError(2));
        error_phi.push_back(resPhi6->GetFunction("fit")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resPhi->SetBinContent(i + 1, sigma_phi[i]);
          resPhi->SetBinError(i + 1, error_phi[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for phi resolution!\n";
        for (int i = 0; i < 6; i++) {
          resPhi->SetBinContent(i + 1, -1);
          resPhi->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for phi resolution cannot be found!\n";
    }

    if (meresVtxZ_eta0to0p7 && meresVtxZ_eta0p7to1 && meresVtxZ_eta1to1p2 && meresVtxZ_eta1p2to1p6 &&
        meresVtxZ_eta1p6to2 && meresVtxZ_eta2to2p4) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resVtxZ_1 = meresVtxZ_eta0to0p7->getTH1F();
      TH1F *resVtxZ_2 = meresVtxZ_eta0p7to1->getTH1F();
      TH1F *resVtxZ_3 = meresVtxZ_eta1to1p2->getTH1F();
      TH1F *resVtxZ_4 = meresVtxZ_eta1p2to1p6->getTH1F();
      TH1F *resVtxZ_5 = meresVtxZ_eta1p6to2->getTH1F();
      TH1F *resVtxZ_6 = meresVtxZ_eta2to2p4->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_VtxZ = ibooker.book1D("VtxZResolution", "VtxZ resolution vs |#eta|", eta_binnum, eta_bins);
      TH1F *resVtxZ = me_res_VtxZ->getTH1F();
      resVtxZ->GetXaxis()->SetTitle("tracking particle |#eta|");
      resVtxZ->GetYaxis()->SetTitle("#sigma(#DeltaVtxZ) [cm]");
      resVtxZ->SetMinimum(0.0);
      resVtxZ->SetStats(false);

      int testNumEntries6 = resVtxZ_1->GetEntries();
      if (testNumEntries6 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resVtxZ_1->Fit(fit3, "R");
        resVtxZ_2->Fit(fit3, "R");
        resVtxZ_3->Fit(fit3, "R");
        resVtxZ_4->Fit(fit3, "R");
        resVtxZ_5->Fit(fit3, "R");
        resVtxZ_6->Fit(fit3, "R");
        sigma_VtxZ.push_back(resVtxZ_1->GetFunction("fit3")->GetParameter(2));
        sigma_VtxZ.push_back(resVtxZ_2->GetFunction("fit3")->GetParameter(2));
        sigma_VtxZ.push_back(resVtxZ_3->GetFunction("fit3")->GetParameter(2));
        sigma_VtxZ.push_back(resVtxZ_4->GetFunction("fit3")->GetParameter(2));
        sigma_VtxZ.push_back(resVtxZ_5->GetFunction("fit3")->GetParameter(2));
        sigma_VtxZ.push_back(resVtxZ_6->GetFunction("fit3")->GetParameter(2));
        error_VtxZ.push_back(resVtxZ_1->GetFunction("fit3")->GetParError(2));
        error_VtxZ.push_back(resVtxZ_2->GetFunction("fit3")->GetParError(2));
        error_VtxZ.push_back(resVtxZ_3->GetFunction("fit3")->GetParError(2));
        error_VtxZ.push_back(resVtxZ_4->GetFunction("fit3")->GetParError(2));
        error_VtxZ.push_back(resVtxZ_5->GetFunction("fit3")->GetParError(2));
        error_VtxZ.push_back(resVtxZ_6->GetFunction("fit3")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resVtxZ->SetBinContent(i + 1, sigma_VtxZ[i]);
          resVtxZ->SetBinError(i + 1, error_VtxZ[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for VtxZ resolution!\n";
        for (int i = 0; i < 6; i++) {
          resVtxZ->SetBinContent(i + 1, -1);
          resVtxZ->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for VtxZ resolution cannot be found!\n";
    }

    if (meresd0_eta0to0p7 && meresd0_eta0p7to1 && meresd0_eta1to1p2 && meresd0_eta1p2to1p6 && meresd0_eta1p6to2 &&
        meresd0_eta2to2p4) {
      // Set the current directoy
      dbe->setCurrentFolder("SiOuterTrackerV/Tracks/FinalResolution");

      // Grab the histograms
      TH1F *resd0_1 = meresd0_eta0to0p7->getTH1F();
      TH1F *resd0_2 = meresd0_eta0p7to1->getTH1F();
      TH1F *resd0_3 = meresd0_eta1to1p2->getTH1F();
      TH1F *resd0_4 = meresd0_eta1p2to1p6->getTH1F();
      TH1F *resd0_5 = meresd0_eta1p6to2->getTH1F();
      TH1F *resd0_6 = meresd0_eta2to2p4->getTH1F();

      // Book the new histogram to contain the results
      MonitorElement *me_res_d0 = ibooker.book1D("d0Resolution", "d_{0} resolution vs |#eta|", eta_binnum, eta_bins);
      TH1F *resd0 = me_res_d0->getTH1F();
      resd0->GetXaxis()->SetTitle("tracking particle |#eta|");
      resd0->GetYaxis()->SetTitle("#sigma(#Deltad_{0}) [cm]");
      resd0->SetMinimum(0.0);
      resd0->SetStats(false);

      int testNumEntries7 = resd0_1->GetEntries();
      if (testNumEntries7 > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        resd0_1->Fit(fit, "R");
        resd0_2->Fit(fit, "R");
        resd0_3->Fit(fit, "R");
        resd0_4->Fit(fit, "R");
        resd0_5->Fit(fit, "R");
        resd0_6->Fit(fit, "R");
        sigma_d0.push_back(resd0_1->GetFunction("fit")->GetParameter(2));
        sigma_d0.push_back(resd0_2->GetFunction("fit")->GetParameter(2));
        sigma_d0.push_back(resd0_3->GetFunction("fit")->GetParameter(2));
        sigma_d0.push_back(resd0_4->GetFunction("fit")->GetParameter(2));
        sigma_d0.push_back(resd0_5->GetFunction("fit")->GetParameter(2));
        sigma_d0.push_back(resd0_6->GetFunction("fit")->GetParameter(2));
        error_d0.push_back(resd0_1->GetFunction("fit")->GetParError(2));
        error_d0.push_back(resd0_2->GetFunction("fit")->GetParError(2));
        error_d0.push_back(resd0_3->GetFunction("fit")->GetParError(2));
        error_d0.push_back(resd0_4->GetFunction("fit")->GetParError(2));
        error_d0.push_back(resd0_5->GetFunction("fit")->GetParError(2));
        error_d0.push_back(resd0_6->GetFunction("fit")->GetParError(2));

        // Fill the new histogram to create resolution plot
        for (int i = 0; i < 6; i++) {
          resd0->SetBinContent(i + 1, sigma_d0[i]);
          resd0->SetBinError(i + 1, error_d0[i]);
        }
      } else {
        edm::LogWarning("DataNotFound") << "L1 tracks not found for d0 resolution!\n";
        for (int i = 0; i < 6; i++) {
          resd0->SetBinContent(i + 1, -1);
          resd0->SetBinError(i + 1, -1);
        }
      }
    }  // if ME found
    else {
      edm::LogWarning("DataNotFound") << "Monitor elements for d0 resolution cannot be found!\n";
    }

  }  // if dbe found
  else {
    edm::LogWarning("DataNotFound") << "Cannot find valid DQM back end \n";
  }
  delete fit;
  delete fit2;
  delete fit3;
}  // end dqmEndJob

DEFINE_FWK_MODULE(OuterTrackerMCHarvester);
