#include "OuterTrackerMCHarvester.h"
#include "TFitResult.h"

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

      if (resPt1a->GetEntries() > 0 && resPt2a->GetEntries() > 0 && resPt3a->GetEntries() > 0 &&
          resPt4a->GetEntries() > 0 && resPt5a->GetEntries() > 0 && resPt6a->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_pt1a = resPt1a->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt2a = resPt2a->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt3a = resPt3a->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt4a = resPt4a->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt5a = resPt5a->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt6a = resPt6a->Fit(fit2, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_pt1a = r_pt1a; Int_t fs_pt2a = r_pt2a; Int_t fs_pt3a = r_pt3a;
        Int_t fs_pt4a = r_pt4a; Int_t fs_pt5a = r_pt5a; Int_t fs_pt6a = r_pt6a;

        if (fs_pt1a==0) {sigma_pt1.push_back(r_pt1a->Parameter(2)); error_pt1.push_back(r_pt1a->ParError(2));}
        else {sigma_pt1.push_back(-1); error_pt1.push_back(-1);}
        if (fs_pt2a==0) {sigma_pt1.push_back(r_pt2a->Parameter(2)); error_pt1.push_back(r_pt2a->ParError(2));}
        else {sigma_pt1.push_back(-1); error_pt1.push_back(-1);}
        if (fs_pt3a==0) {sigma_pt1.push_back(r_pt3a->Parameter(2)); error_pt1.push_back(r_pt3a->ParError(2));}
        else {sigma_pt1.push_back(-1); error_pt1.push_back(-1);}
        if (fs_pt4a==0) {sigma_pt1.push_back(r_pt4a->Parameter(2)); error_pt1.push_back(r_pt4a->ParError(2));}
        else {sigma_pt1.push_back(-1); error_pt1.push_back(-1);}
        if (fs_pt5a==0) {sigma_pt1.push_back(r_pt5a->Parameter(2)); error_pt1.push_back(r_pt5a->ParError(2));}
        else {sigma_pt1.push_back(-1); error_pt1.push_back(-1);}
        if (fs_pt6a==0) {sigma_pt1.push_back(r_pt6a->Parameter(2)); error_pt1.push_back(r_pt6a->ParError(2));}
        else {sigma_pt1.push_back(-1); error_pt1.push_back(-1);}

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

      if (resPt1b->GetEntries() > 0 && resPt2b->GetEntries() > 0 && resPt3b->GetEntries() > 0 &&
          resPt4b->GetEntries() > 0 && resPt5b->GetEntries() > 0 && resPt6b->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_pt1b = resPt1b->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt2b = resPt2b->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt3b = resPt3b->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt4b = resPt4b->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt5b = resPt5b->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt6b = resPt6b->Fit(fit2, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_pt1b = r_pt1b; Int_t fs_pt2b = r_pt2b; Int_t fs_pt3b = r_pt3b;
        Int_t fs_pt4b = r_pt4b; Int_t fs_pt5b = r_pt5b; Int_t fs_pt6b = r_pt6b;

        if (fs_pt1b==0) {sigma_pt2.push_back(r_pt1b->Parameter(2)); error_pt2.push_back(r_pt1b->ParError(2));}
        else {sigma_pt2.push_back(-1); error_pt2.push_back(-1);}
        if (fs_pt2b==0) {sigma_pt2.push_back(r_pt2b->Parameter(2)); error_pt2.push_back(r_pt2b->ParError(2));}
        else {sigma_pt2.push_back(-1); error_pt2.push_back(-1);}
        if (fs_pt3b==0) {sigma_pt2.push_back(r_pt3b->Parameter(2)); error_pt2.push_back(r_pt3b->ParError(2));}
        else {sigma_pt2.push_back(-1); error_pt2.push_back(-1);}
        if (fs_pt4b==0) {sigma_pt2.push_back(r_pt4b->Parameter(2)); error_pt2.push_back(r_pt4b->ParError(2));}
        else {sigma_pt2.push_back(-1); error_pt2.push_back(-1);}
        if (fs_pt5b==0) {sigma_pt2.push_back(r_pt5b->Parameter(2)); error_pt2.push_back(r_pt5b->ParError(2));}
        else {sigma_pt2.push_back(-1); error_pt2.push_back(-1);}
        if (fs_pt6b==0) {sigma_pt2.push_back(r_pt6b->Parameter(2)); error_pt2.push_back(r_pt6b->ParError(2));}
        else {sigma_pt2.push_back(-1); error_pt2.push_back(-1);}

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

      if (resPt1c->GetEntries() > 0 && resPt2c->GetEntries() > 0 && resPt3c->GetEntries() > 0 &&
          resPt4c->GetEntries() > 0 && resPt5c->GetEntries() > 0 && resPt6c->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_pt1c = resPt1c->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt2c = resPt2c->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt3c = resPt3c->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt4c = resPt4c->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt5c = resPt5c->Fit(fit2, "Q", "RS");
        TFitResultPtr r_pt6c = resPt6c->Fit(fit2, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_pt1c = r_pt1c; Int_t fs_pt2c = r_pt2c; Int_t fs_pt3c = r_pt3c;
        Int_t fs_pt4c = r_pt4c; Int_t fs_pt5c = r_pt5c; Int_t fs_pt6c = r_pt6c;

        if (fs_pt1c==0) {sigma_pt3.push_back(r_pt1c->Parameter(2)); error_pt3.push_back(r_pt1c->ParError(2));}
        else {sigma_pt3.push_back(-1); error_pt3.push_back(-1);}
        if (fs_pt2c==0) {sigma_pt3.push_back(r_pt2c->Parameter(2)); error_pt3.push_back(r_pt2c->ParError(2));}
        else {sigma_pt3.push_back(-1); error_pt3.push_back(-1);}
        if (fs_pt3c==0) {sigma_pt3.push_back(r_pt3c->Parameter(2)); error_pt3.push_back(r_pt3c->ParError(2));}
        else {sigma_pt3.push_back(-1); error_pt3.push_back(-1);}
        if (fs_pt4c==0) {sigma_pt3.push_back(r_pt4c->Parameter(2)); error_pt3.push_back(r_pt4c->ParError(2));}
        else {sigma_pt3.push_back(-1); error_pt3.push_back(-1);}
        if (fs_pt5c==0) {sigma_pt3.push_back(r_pt5c->Parameter(2)); error_pt3.push_back(r_pt5c->ParError(2));}
        else {sigma_pt3.push_back(-1); error_pt3.push_back(-1);}
        if (fs_pt6c==0) {sigma_pt3.push_back(r_pt6c->Parameter(2)); error_pt3.push_back(r_pt6c->ParError(2));}
        else {sigma_pt3.push_back(-1); error_pt3.push_back(-1);}

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

      if (resEta1->GetEntries() > 0 && resEta2->GetEntries() > 0 && resEta3->GetEntries() > 0 &&
          resEta4->GetEntries() > 0 && resEta5->GetEntries() > 0 && resEta6->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_eta1 = resEta1->Fit(fit, "Q", "RS");
        TFitResultPtr r_eta2 = resEta2->Fit(fit, "Q", "RS");
        TFitResultPtr r_eta3 = resEta3->Fit(fit, "Q", "RS");
        TFitResultPtr r_eta4 = resEta4->Fit(fit, "Q", "RS");
        TFitResultPtr r_eta5 = resEta5->Fit(fit, "Q", "RS");
        TFitResultPtr r_eta6 = resEta6->Fit(fit, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_eta1 = r_eta1; Int_t fs_eta2 = r_eta2; Int_t fs_eta3 = r_eta3;
        Int_t fs_eta4 = r_eta4; Int_t fs_eta5 = r_eta5; Int_t fs_eta6 = r_eta6;

        if (fs_eta1==0) {sigma_eta.push_back(r_eta1->Parameter(2)); error_eta.push_back(r_eta1->ParError(2));}
        else {sigma_eta.push_back(-1); error_eta.push_back(-1);}
        if (fs_eta2==0) {sigma_eta.push_back(r_eta2->Parameter(2)); error_eta.push_back(r_eta2->ParError(2));}
        else {sigma_eta.push_back(-1); error_eta.push_back(-1);}
        if (fs_eta3==0) {sigma_eta.push_back(r_eta3->Parameter(2)); error_eta.push_back(r_eta3->ParError(2));}
        else {sigma_eta.push_back(-1); error_eta.push_back(-1);}
        if (fs_eta4==0) {sigma_eta.push_back(r_eta4->Parameter(2)); error_eta.push_back(r_eta4->ParError(2));}
        else {sigma_eta.push_back(-1); error_eta.push_back(-1);}
        if (fs_eta5==0) {sigma_eta.push_back(r_eta5->Parameter(2)); error_eta.push_back(r_eta5->ParError(2));}
        else {sigma_eta.push_back(-1); error_eta.push_back(-1);}
        if (fs_eta6==0) {sigma_eta.push_back(r_eta6->Parameter(2)); error_eta.push_back(r_eta6->ParError(2));}
        else {sigma_eta.push_back(-1); error_eta.push_back(-1);}

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

      if (resPhi1->GetEntries() > 0 && resPhi2->GetEntries() > 0 && resPhi3->GetEntries() > 0 &&
          resPhi4->GetEntries() > 0 && resPhi5->GetEntries() > 0 && resPhi6->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_phi1 = resPhi1->Fit(fit, "Q", "RS");
        TFitResultPtr r_phi2 = resPhi2->Fit(fit, "Q", "RS");
        TFitResultPtr r_phi3 = resPhi3->Fit(fit, "Q", "RS");
        TFitResultPtr r_phi4 = resPhi4->Fit(fit, "Q", "RS");
        TFitResultPtr r_phi5 = resPhi5->Fit(fit, "Q", "RS");
        TFitResultPtr r_phi6 = resPhi6->Fit(fit, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_phi1 = r_phi1; Int_t fs_phi2 = r_phi2; Int_t fs_phi3 = r_phi3;
        Int_t fs_phi4 = r_phi4; Int_t fs_phi5 = r_phi5; Int_t fs_phi6 = r_phi6;

        if (fs_phi1==0) {sigma_phi.push_back(r_phi1->Parameter(2)); error_phi.push_back(r_phi1->ParError(2));}
        else {sigma_phi.push_back(-1); error_phi.push_back(-1);}
        if (fs_phi2==0) {sigma_phi.push_back(r_phi2->Parameter(2)); error_phi.push_back(r_phi2->ParError(2));}
        else {sigma_phi.push_back(-1); error_phi.push_back(-1);}
        if (fs_phi3==0) {sigma_phi.push_back(r_phi3->Parameter(2)); error_phi.push_back(r_phi3->ParError(2));}
        else {sigma_phi.push_back(-1); error_phi.push_back(-1);}
        if (fs_phi4==0) {sigma_phi.push_back(r_phi4->Parameter(2)); error_phi.push_back(r_phi4->ParError(2));}
        else {sigma_phi.push_back(-1); error_phi.push_back(-1);}
        if (fs_phi5==0) {sigma_phi.push_back(r_phi5->Parameter(2)); error_phi.push_back(r_phi5->ParError(2));}
        else {sigma_phi.push_back(-1); error_phi.push_back(-1);}
        if (fs_phi6==0) {sigma_phi.push_back(r_phi6->Parameter(2)); error_phi.push_back(r_phi6->ParError(2));}
        else {sigma_phi.push_back(-1); error_phi.push_back(-1);}

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

      if (resVtxZ_1->GetEntries() > 0 && resVtxZ_2->GetEntries() > 0 && resVtxZ_3->GetEntries() > 0 &&
          resVtxZ_4->GetEntries() > 0 && resVtxZ_5->GetEntries() > 0 && resVtxZ_6->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_vtxz1 = resVtxZ_1->Fit(fit3, "Q", "RS");
        TFitResultPtr r_vtxz2 = resVtxZ_2->Fit(fit3, "Q", "RS");
        TFitResultPtr r_vtxz3 = resVtxZ_3->Fit(fit3, "Q", "RS");
        TFitResultPtr r_vtxz4 = resVtxZ_4->Fit(fit3, "Q", "RS");
        TFitResultPtr r_vtxz5 = resVtxZ_5->Fit(fit3, "Q", "RS");
        TFitResultPtr r_vtxz6 = resVtxZ_6->Fit(fit3, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_vtxz1 = r_vtxz1; Int_t fs_vtxz2 = r_vtxz2; Int_t fs_vtxz3 = r_vtxz3;
        Int_t fs_vtxz4 = r_vtxz4; Int_t fs_vtxz5 = r_vtxz5; Int_t fs_vtxz6 = r_vtxz6;

        if (fs_vtxz1==0) {sigma_VtxZ.push_back(r_vtxz1->Parameter(2)); error_VtxZ.push_back(r_vtxz1->ParError(2));}
        else {sigma_VtxZ.push_back(-1); error_VtxZ.push_back(-1);}
        if (fs_vtxz2==0) {sigma_VtxZ.push_back(r_vtxz2->Parameter(2)); error_VtxZ.push_back(r_vtxz2->ParError(2));}
        else {sigma_VtxZ.push_back(-1); error_VtxZ.push_back(-1);}
        if (fs_vtxz3==0) {sigma_VtxZ.push_back(r_vtxz3->Parameter(2)); error_VtxZ.push_back(r_vtxz3->ParError(2));}
        else {sigma_VtxZ.push_back(-1); error_VtxZ.push_back(-1);}
        if (fs_vtxz4==0) {sigma_VtxZ.push_back(r_vtxz4->Parameter(2)); error_VtxZ.push_back(r_vtxz4->ParError(2));}
        else {sigma_VtxZ.push_back(-1); error_VtxZ.push_back(-1);}
        if (fs_vtxz5==0) {sigma_VtxZ.push_back(r_vtxz5->Parameter(2)); error_VtxZ.push_back(r_vtxz5->ParError(2));}
        else {sigma_VtxZ.push_back(-1); error_VtxZ.push_back(-1);}
        if (fs_vtxz6==0) {sigma_VtxZ.push_back(r_vtxz6->Parameter(2)); error_VtxZ.push_back(r_vtxz6->ParError(2));}
        else {sigma_VtxZ.push_back(-1); error_VtxZ.push_back(-1);}

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

      if (resd0_1->GetEntries() > 0 && resd0_2->GetEntries() > 0 && resd0_3->GetEntries() > 0 &&
          resd0_4->GetEntries() > 0 && resd0_5->GetEntries() > 0 && resd0_6->GetEntries() > 0) {
        // Fit the histograms with a gaussian curve - take sigma and the error
        // from the fit
        TFitResultPtr r_d01 = resd0_1->Fit(fit, "Q", "RS");
        TFitResultPtr r_d02 = resd0_2->Fit(fit, "Q", "RS");
        TFitResultPtr r_d03 = resd0_3->Fit(fit, "Q", "RS");
        TFitResultPtr r_d04 = resd0_4->Fit(fit, "Q", "RS");
        TFitResultPtr r_d05 = resd0_5->Fit(fit, "Q", "RS");
        TFitResultPtr r_d06 = resd0_6->Fit(fit, "Q", "RS");

        //Check fit status before extracting parameters
        Int_t fs_d01 = r_d01; Int_t fs_d02 = r_d02; Int_t fs_d03 = r_d03;
        Int_t fs_d04 = r_d04; Int_t fs_d05 = r_d05; Int_t fs_d06 = r_d06;

        if (fs_d01==0) {sigma_d0.push_back(r_d01->Parameter(2)); error_d0.push_back(r_d01->ParError(2));}
        else {sigma_d0.push_back(-1); error_d0.push_back(-1);}
        if (fs_d02==0) {sigma_d0.push_back(r_d02->Parameter(2)); error_d0.push_back(r_d02->ParError(2));}
        else {sigma_d0.push_back(-1); error_d0.push_back(-1);}
        if (fs_d03==0) {sigma_d0.push_back(r_d03->Parameter(2)); error_d0.push_back(r_d03->ParError(2));}
        else {sigma_d0.push_back(-1); error_d0.push_back(-1);}
        if (fs_d04==0) {sigma_d0.push_back(r_d04->Parameter(2)); error_d0.push_back(r_d04->ParError(2));}
        else {sigma_d0.push_back(-1); error_d0.push_back(-1);}
        if (fs_d05==0) {sigma_d0.push_back(r_d05->Parameter(2)); error_d0.push_back(r_d05->ParError(2));}
        else {sigma_d0.push_back(-1); error_d0.push_back(-1);}
        if (fs_d06==0) {sigma_d0.push_back(r_d06->Parameter(2)); error_d0.push_back(r_d06->ParError(2));}
        else {sigma_d0.push_back(-1); error_d0.push_back(-1);}

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
