// -*- C++ -*-
//
// Package:    HcalDigisValidation
// Class:      HcalDigisValidation
//
/**\class HcalDigisValidation HcalDigisValidation.cc Validation/HcalDigis/src/HcalDigisValidation.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
 */
//
// Original Author:  Ali Fahim,22 R-013,+41227672649,
//         Created:  Wed Mar 23 11:42:34 CET 2011
//
//

#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include <Validation/HcalDigis/interface/HcalDigisValidation.h>
#include "FWCore/Framework/interface/MakerMacros.h"

HcalDigisValidation::HcalDigisValidation(const edm::ParameterSet& iConfig) {

    using namespace std;

    subdet_ = iConfig.getUntrackedParameter<std::string > ("subdetector", "all");
    outputFile_ = iConfig.getUntrackedParameter<std::string > ("outputFile", "");
    inputTag_ = iConfig.getParameter<edm::InputTag > ("digiLabel");
    mc_ = iConfig.getUntrackedParameter<std::string > ("mc", "no");
    mode_ = iConfig.getUntrackedParameter<std::string > ("mode", "multi");
    dirName_ = iConfig.getUntrackedParameter<std::string > ("dirName", "HcalDigisV/HcalDigiTask");

    // register for data access
    if (iConfig.exists("simHits"))
    {
	tok_mc_ = consumes<edm::PCaloHitContainer>(iConfig.getUntrackedParameter<edm::InputTag>("simHits"));
    }
    tok_hbhe_ = consumes<edm::SortedCollection<HBHEDataFrame> >(inputTag_);
    tok_ho_ = consumes<edm::SortedCollection<HODataFrame> >(inputTag_);
    tok_hf_ = consumes<edm::SortedCollection<HFDataFrame> >(inputTag_);
    tok_emulTPs_ = consumes<HcalTrigPrimDigiCollection>(edm::InputTag("emulDigis"));
    tok_dataTPs_ = consumes<HcalTrigPrimDigiCollection>(edm::InputTag("simHcalTriggerPrimitiveDigis"));

    nevent1 = 0;
    nevent2 = 0;
    nevent3 = 0;
    nevent4 = 0;
    nevtot = 0;

    msm_ = new std::map<std::string, MonitorElement*>();

    if (outputFile_.size() != 0) edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
    else edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will NOT be saved";

}


HcalDigisValidation::~HcalDigisValidation() {
    delete msm_;
}


void HcalDigisValidation::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es)
{

    ib.setCurrentFolder(dirName_);

    // book
    book1D(ib,"nevtot", 1, 0, 1);
    int bnoise = 0;
    int bmc = 0;
    if (subdet_ == "noise") bnoise = 1;
    if (mc_ == "yes") bmc = 1;
    if (subdet_ == "noise" || subdet_ == "all") {
        booking(ib,"HB", bnoise, bmc);
        booking(ib,"HO", bnoise, bmc);
        booking(ib,"HF", bnoise, bmc);
        booking(ib,"HE", bnoise, bmc);
    } else {
        booking(ib,subdet_, 0, bmc);
    }

    
    HistLim tp_hl_et(260, -10, 250);
    HistLim tp_hl_ntp(640, -20, 3180);
    HistLim tp_hl_ntp_sub(404, -20, 2000);
    HistLim tp_hl_ieta(85, -42.5, 42.5);
    

    book1D(ib,"HcalDigiTask_tp_et", tp_hl_et);
    book1D(ib,"HcalDigiTask_tp_et_HB", tp_hl_et);
    book1D(ib,"HcalDigiTask_tp_et_HE", tp_hl_et);
    book1D(ib,"HcalDigiTask_tp_et_HF", tp_hl_et);
    book1D(ib,"HcalDigiTask_tp_ntp", tp_hl_ntp);
    book1D(ib,"HcalDigiTask_tp_ntp_HB", tp_hl_ntp_sub);
    book1D(ib,"HcalDigiTask_tp_ntp_HE", tp_hl_ntp_sub);
    book1D(ib,"HcalDigiTask_tp_ntp_HF", tp_hl_ntp_sub);
    book1D(ib,"HcalDigiTask_tp_ntp_ieta", tp_hl_ieta);
    book1D(ib,"HcalDigiTask_tp_ntp_10_ieta", tp_hl_ieta);
    book2D(ib,"HcalDigiTask_tp_et_ieta", tp_hl_ieta, tp_hl_et);
    bookPf(ib,"HcalDigiTask_tp_ave_et_ieta", tp_hl_ieta, tp_hl_et, " "); 

}

void HcalDigisValidation::booking(DQMStore::IBooker &ib, const std::string bsubdet, int bnoise, int bmc) {

    // Adjust/Optimize binning (JR Dittmann, 16-JUL-2015)

    HistLim Ndigis(2600, 0., 2600.);
    HistLim ndigis(520, -20., 1020.);
    HistLim sime(200, 0., 1.0);

    HistLim digiAmp(360, -100., 7100.);
    HistLim ratio(2000, -100., 3900.);
    HistLim sumAmp(100, -500., 1500.);

    HistLim nbin(10, 0., 10.);

    HistLim pedestal(80, -1.0, 15.);
    HistLim pedestalfC(400, -10., 30.);

    HistLim frac(80, -0.20, 1.40);

    HistLim pedLim(80, 0., 8.);
    HistLim pedWidthLim(100, 0., 2.);

    HistLim gainLim(120, 0., 0.6);
    HistLim gainWidthLim(160, 0., 0.32);

    HistLim ietaLim(85, -42.5, 42.5);
    HistLim iphiLim(74, -0.5, 73.5);

    if (bsubdet == "HE") {
        sime = HistLim(200, 0., 1.0);
    } else if (bsubdet == "HF") {
        sime = HistLim(100, 0., 100.);
        pedLim = HistLim(100, 0., 20.);
        pedWidthLim = HistLim(100, 0., 5.);
        frac = HistLim(400, -4.00, 4.00);
    } else if (bsubdet == "HO") {
        sime = HistLim(200, 0., 1.0);
        gainLim = HistLim(160, 0., 1.6);
    }

    Char_t histo[100];
    const char * sub = bsubdet.c_str();
    if (bnoise == 0) {
        // number of digis in each subdetector
        sprintf(histo, "HcalDigiTask_Ndigis_%s", sub);
        book1D(ib, histo, Ndigis);

        // maps of occupancies
        sprintf(histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth1_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);

        sprintf(histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth2_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);

        sprintf(histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth3_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);

        sprintf(histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth4_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);

        // occupancies vs ieta
        sprintf(histo, "HcalDigiTask_occupancy_vs_ieta_depth1_%s", sub);
        book1D(ib, histo, ietaLim);

        sprintf(histo, "HcalDigiTask_occupancy_vs_ieta_depth2_%s", sub);
        book1D(ib, histo, ietaLim);

        sprintf(histo, "HcalDigiTask_occupancy_vs_ieta_depth3_%s", sub);
        book1D(ib, histo, ietaLim);

        sprintf(histo, "HcalDigiTask_occupancy_vs_ieta_depth4_%s", sub);
        book1D(ib, histo, ietaLim);


        // maps of sum of amplitudes (sum lin.digis(4,5,6,7) - ped) all depths
/*
        sprintf(histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth1_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth2_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth3_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth4_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
*/
        // just 1D of all cells' amplitudes
        sprintf(histo, "HcalDigiTask_sum_all_amplitudes_%s", sub);
        book1D(ib, histo, sumAmp);

        sprintf(histo, "HcalDigiTask_number_of_amplitudes_above_10fC_%s", sub);
        book1D(ib, histo, ndigis);

        sprintf(histo, "HcalDigiTask_ADC0_adc_depth1_%s", sub);
        book1D(ib, histo, pedestal);
        sprintf(histo, "HcalDigiTask_ADC0_adc_depth2_%s", sub);
        book1D(ib, histo, pedestal);
        sprintf(histo, "HcalDigiTask_ADC0_adc_depth3_%s", sub);
        book1D(ib, histo, pedestal);
        sprintf(histo, "HcalDigiTask_ADC0_adc_depth4_%s", sub);
        book1D(ib, histo, pedestal);

        sprintf(histo, "HcalDigiTask_ADC0_fC_depth1_%s", sub);
        book1D(ib, histo, pedestalfC);
        sprintf(histo, "HcalDigiTask_ADC0_fC_depth2_%s", sub);
        book1D(ib, histo, pedestalfC);
        sprintf(histo, "HcalDigiTask_ADC0_fC_depth3_%s", sub);
        book1D(ib, histo, pedestalfC);
        sprintf(histo, "HcalDigiTask_ADC0_fC_depth4_%s", sub);
        book1D(ib, histo, pedestalfC);

        sprintf(histo, "HcalDigiTask_signal_amplitude_%s", sub);
        book1D(ib, histo, digiAmp);
        sprintf(histo, "HcalDigiTask_signal_amplitude_depth1_%s", sub);
        book1D(ib, histo, digiAmp);
        sprintf(histo, "HcalDigiTask_signal_amplitude_depth2_%s", sub);
        book1D(ib, histo, digiAmp);
        sprintf(histo, "HcalDigiTask_signal_amplitude_depth3_%s", sub);
        book1D(ib, histo, digiAmp);
        sprintf(histo, "HcalDigiTask_signal_amplitude_depth4_%s", sub);
        book1D(ib, histo, digiAmp);

        sprintf(histo, "HcalDigiTask_signal_amplitude_vs_bin_all_depths_%s", sub);
        book2D(ib, histo, nbin, digiAmp);

/*
        sprintf(histo, "HcalDigiTask_all_amplitudes_vs_bin_depth1_%s", sub);
        book2D(ib, histo, nbin, digiAmp);
        sprintf(histo, "HcalDigiTask_all_amplitudes_vs_bin_depth2_%s", sub);
        book2D(ib, histo, nbin, digiAmp);
*/
        sprintf(histo, "HcalDigiTask_all_amplitudes_vs_bin_1D_depth1_%s", sub);
        book1D(ib, histo, nbin);
        sprintf(histo, "HcalDigiTask_all_amplitudes_vs_bin_1D_depth2_%s", sub);
        book1D(ib, histo, nbin);

        sprintf(histo, "HcalDigiTask_bin_5_frac_%s", sub);
        book1D(ib, histo, frac);
        sprintf(histo, "HcalDigiTask_bin_6_7_frac_%s", sub);
        book1D(ib, histo, frac);

        if (bmc == 1) {
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_%s", sub);
            book2D(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_depth1_%s", sub);
            book2D(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_depth2_%s", sub);
            book2D(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_depth3_%s", sub);
            book2D(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_depth4_%s", sub);
            book2D(ib, histo, sime, digiAmp);

            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_%s", sub);
            bookPf(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth1_%s", sub);
            bookPf(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth2_%s", sub);
            bookPf(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth3_%s", sub);
            bookPf(ib, histo, sime, digiAmp);
            sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth4_%s", sub);
            bookPf(ib, histo, sime, digiAmp);

            sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_%s", sub);
            book1D(ib, histo, ratio);
            sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth1_%s", sub);
            book1D(ib, histo, ratio);
            sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth2_%s", sub);
            book1D(ib, histo, ratio);
            sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth3_%s", sub);
            book1D(ib, histo, ratio);
            sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth4_%s", sub);
            book1D(ib, histo, ratio);
        }//mc only

    } else { // noise only

        // EVENT "1" distributions of all cells properties


        if (subdet_ == "HB" || subdet_ == "HE" || subdet_ == "HF") {
            sprintf(histo, "HcalDigiTask_gain_capId0_Depth1_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId1_Depth1_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId2_Depth1_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId3_Depth1_%s", sub);
            book1D(ib, histo, gainLim);

            sprintf(histo, "HcalDigiTask_gain_capId0_Depth2_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId1_Depth2_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId2_Depth2_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId3_Depth2_%s", sub);
            book1D(ib, histo, gainLim);

            sprintf(histo, "HcalDigiTask_gainWidth_capId0_Depth1_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId1_Depth1_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId2_Depth1_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId3_Depth1_%s", sub);
            book1D(ib, histo, gainWidthLim);

            sprintf(histo, "HcalDigiTask_gainWidth_capId0_Depth2_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId1_Depth2_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId2_Depth2_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId3_Depth2_%s", sub);
            book1D(ib, histo, gainWidthLim);

            sprintf(histo, "HcalDigiTask_pedestal_capId0_Depth1_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId1_Depth1_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId2_Depth1_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId3_Depth1_%s", sub);
            book1D(ib, histo, pedLim);

            sprintf(histo, "HcalDigiTask_pedestal_capId0_Depth2_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId1_Depth2_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId2_Depth2_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId3_Depth2_%s", sub);
            book1D(ib, histo, pedLim);

            sprintf(histo, "HcalDigiTask_pedestal_width_capId0_Depth1_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId1_Depth1_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId2_Depth1_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId3_Depth1_%s", sub);
            book1D(ib, histo, pedWidthLim);

            sprintf(histo, "HcalDigiTask_pedestal_width_capId0_Depth2_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId1_Depth2_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId2_Depth2_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId3_Depth2_%s", sub);
            book1D(ib, histo, pedWidthLim);

        }

        if (subdet_ == "HE") {
            sprintf(histo, "HcalDigiTask_gain_capId0_Depth3_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId1_Depth3_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId2_Depth3_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId3_Depth3_%s", sub);
            book1D(ib, histo, gainLim);

            sprintf(histo, "HcalDigiTask_gainWidth_capId0_Depth3_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId1_Depth3_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId2_Depth3_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId3_Depth3_%s", sub);
            book1D(ib, histo, gainWidthLim);

            sprintf(histo, "HcalDigiTask_pedestal_capId0_Depth3_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId1_Depth3_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId2_Depth3_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId3_Depth3_%s", sub);
            book1D(ib, histo, pedLim);

            sprintf(histo, "HcalDigiTask_pedestal_width_capId0_Depth3_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId1_Depth3_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId2_Depth3_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId3_Depth3_%s", sub);
            book1D(ib, histo, pedWidthLim);

        }

        if (subdet_ == "HO") {
            sprintf(histo, "HcalDigiTask_gain_capId0_Depth4_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId1_Depth4_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId2_Depth4_%s", sub);
            book1D(ib, histo, gainLim);
            sprintf(histo, "HcalDigiTask_gain_capId3_Depth4_%s", sub);
            book1D(ib, histo, gainLim);

            sprintf(histo, "HcalDigiTask_gainWidth_capId0_Depth4_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId1_Depth4_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId2_Depth4_%s", sub);
            book1D(ib, histo, gainWidthLim);
            sprintf(histo, "HcalDigiTask_gainWidth_capId3_Depth4_%s", sub);
            book1D(ib, histo, gainWidthLim);


            sprintf(histo, "HcalDigiTask_pedestal_capId0_Depth4_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId1_Depth4_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId2_Depth4_%s", sub);
            book1D(ib, histo, pedLim);
            sprintf(histo, "HcalDigiTask_pedestal_capId3_Depth4_%s", sub);
            book1D(ib, histo, pedLim);

            sprintf(histo, "HcalDigiTask_pedestal_width_capId0_Depth4_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId1_Depth4_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId2_Depth4_%s", sub);
            book1D(ib, histo, pedWidthLim);
            sprintf(histo, "HcalDigiTask_pedestal_width_capId3_Depth4_%s", sub);
            book1D(ib, histo, pedWidthLim);

        }

        sprintf(histo, "HcalDigiTask_gainMap_Depth1_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_gainMap_Depth2_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_gainMap_Depth3_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_gainMap_Depth4_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);

        sprintf(histo, "HcalDigiTask_pwidthMap_Depth1_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_pwidthMap_Depth2_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_pwidthMap_Depth3_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);
        sprintf(histo, "HcalDigiTask_pwidthMap_Depth4_%s", sub);
        book2D(ib, histo, ietaLim, iphiLim);

    } //end of noise-only
}//book

void HcalDigisValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;
    using namespace std;

    iSetup.get<CaloGeometryRecord > ().get(geometry);
    iSetup.get<HcalDbRecord > ().get(conditions);

    //TP Code
    ESHandle<CaloTPGTranscoder> decoder;
    iSetup.get<CaloTPGRecord>().get(decoder);

    ESHandle<HcalTrigTowerGeometry> tp_geometry;
    iSetup.get<CaloGeometryRecord>().get(tp_geometry);

    iSetup.get<HcalRecNumberingRecord>().get(htopo);

    //Get all handles
    edm::Handle<HcalTrigPrimDigiCollection> emulTPs;
    iEvent.getByToken(tok_emulTPs_, emulTPs);

    edm::Handle<HcalTrigPrimDigiCollection> dataTPs;
    iEvent.getByToken(tok_dataTPs_, dataTPs);
    //iEvent.getByLabel("hcalDigis", dataTPs);

    //~TP Code

    //  std::cout << " >>>>> HcalDigiTester::analyze  hcalselector = "
    //	    << subdet_ << std::endl;

    if (subdet_ != "all") {
       noise_ = 0;
       if (subdet_ == "HB") reco<HBHEDataFrame > (iEvent, iSetup, tok_hbhe_);
       if (subdet_ == "HE") reco<HBHEDataFrame > (iEvent, iSetup, tok_hbhe_);
       if (subdet_ == "HO") reco<HODataFrame > (iEvent, iSetup, tok_ho_);
       if (subdet_ == "HF") reco<HFDataFrame > (iEvent, iSetup, tok_hf_);

        if (subdet_ == "noise") {
            noise_ = 1;
            //      std::cout << " >>>>> HcalDigiTester::analyze  entering noise "
            //	    << std::endl;
    	    subdet_ = "HB";
            reco<HBHEDataFrame > (iEvent, iSetup, tok_hbhe_);
            subdet_ = "HE";
            reco<HBHEDataFrame > (iEvent, iSetup, tok_hbhe_);
            subdet_ = "HO";
            reco<HODataFrame > (iEvent, iSetup, tok_ho_);
            subdet_ = "HF";
            reco<HFDataFrame > (iEvent, iSetup, tok_hf_);
            subdet_ = "noise";
            }
        }// all subdetectors
    else {
        noise_ = 0;

        subdet_ = "HB";
        reco<HBHEDataFrame > (iEvent, iSetup, tok_hbhe_);
        subdet_ = "HE";
        reco<HBHEDataFrame > (iEvent, iSetup, tok_hbhe_);
        subdet_ = "HO";
        reco<HODataFrame > (iEvent, iSetup, tok_ho_);
        subdet_ = "HF";
        reco<HFDataFrame > (iEvent, iSetup, tok_hf_);
        subdet_ = "all";
    }

    fill1D("nevtot", 0);
    nevtot++;

   //TP Code
   //Counters
   int c = 0, chb = 0, che = 0, chf = 0;

   for (HcalTrigPrimDigiCollection::const_iterator itr = dataTPs->begin(); itr != dataTPs->end(); ++itr) {
     int ieta  = itr->id().ieta();

     HcalSubdetector subdet = (HcalSubdetector) 0;
     if      ( abs(ieta) <= 16 )
        subdet = HcalSubdetector::HcalBarrel ;
     else if ( abs(ieta) < tp_geometry->firstHFTower(itr->id().version()) ) 
        subdet = HcalSubdetector::HcalEndcap ;
     else if ( abs(ieta) <= 42 )
        subdet = HcalSubdetector::HcalForward;
     
     /*     HcalSubdetector subdet = (HcalSubdetector) itr->id().subdet(); */

     float en = decoder->hcaletValue(itr->id(), itr->t0());
     
     if (en < 0.00001) continue;

     //Plot the variables

     fill1D("HcalDigiTask_tp_et",en);
     fill2D("HcalDigiTask_tp_et_ieta",ieta,en);
     fillPf("HcalDigiTask_tp_ave_et_ieta",ieta,en);

     ++c;
     if ( subdet == HcalSubdetector::HcalBarrel ) {
        fill1D("HcalDigiTask_tp_et_HB",en);
       ++chb;
     }
     if ( subdet == HcalSubdetector::HcalEndcap ) {
       fill1D("HcalDigiTask_tp_et_HE",en);
       ++che;
     }
     if ( subdet == HcalSubdetector::HcalForward ) {
       fill1D("HcalDigiTask_tp_et_HF",en);
       ++chf;
     }

     fill1D("HcalDigiTask_tp_ntp_ieta",ieta);
     if ( en > 10. ) fill1D("HcalDigiTask_tp_ntp_10_ieta",ieta);

   }//end data TP collection 
   
   fill1D("HcalDigiTask_tp_ntp",c);
   fill1D("HcalDigiTask_tp_ntp_HB",chb);
   fill1D("HcalDigiTask_tp_ntp_HE",che);
   fill1D("HcalDigiTask_tp_ntp_HF",chf);

    //~TP Code
}

template<class Digi> void HcalDigisValidation::reco(const edm::Event& iEvent, const edm::EventSetup& iSetup, const edm::EDGetTokenT<edm::SortedCollection<Digi> > & tok) {


    // HistLim =============================================================

    std::string strtmp;

    // ======================================================================
    using namespace edm;
    typename edm::Handle<edm::SortedCollection<Digi> > digiCollection;
    typename edm::SortedCollection<Digi>::const_iterator digiItr;

    // ADC2fC
    HcalCalibrations calibrations;
    CaloSamples tool;
    iEvent.getByToken(tok, digiCollection);
//    std::cout << "***************RECO*****************" << std::endl;
    int isubdet = 0;
    if (subdet_ == "HB") isubdet = 1;
    if (subdet_ == "HE") isubdet = 2;
    if (subdet_ == "HO") isubdet = 3;
    if (subdet_ == "HF") isubdet = 4;

    if (isubdet == 1) nevent1++;
    if (isubdet == 2) nevent2++;
    if (isubdet == 3) nevent3++;
    if (isubdet == 4) nevent4++;

    int indigis = 0;
    //  amplitude for signal cell at diff. depths
    double ampl1_c = 0.;
    double ampl2_c = 0.;
    double ampl3_c = 0.;
    double ampl4_c = 0.;
    double ampl_c = 0.;

    // is set to 1 if "seed" SimHit is found
    int seedSimHit = 0;

    //  std::cout << " HcalDigiTester::reco :  "
    // 	    << "subdet=" << subdet << "  noise="<< noise_ << std::endl;

    int ieta_Sim = 9999;
    int iphi_Sim = 9999;
    double emax_Sim = -9999.;


    // SimHits MC only
    if (mc_ == "yes") {
        edm::Handle<edm::PCaloHitContainer> hcalHits;
        iEvent.getByToken(tok_mc_, hcalHits);
        const edm::PCaloHitContainer * simhitResult = hcalHits.product();

        if (isubdet != 0 && noise_ == 0) { // signal only SimHits

            for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end(); ++simhits) {

                HcalDetId cell(simhits->id());
                double en = simhits->energy();
                int sub = cell.subdet();
                int ieta = cell.ieta();
                //REMOVED (JRD)  if (ieta > 0) ieta--;
                //REMOVED (JRD)  int iphi = cell.iphi() - 1;
                int iphi = cell.iphi();

                if (en > emax_Sim && sub == isubdet) {
                    emax_Sim = en;
                    ieta_Sim = ieta;
                    iphi_Sim = iphi;
                    // to limit "seed" SimHit energy in case of "multi" event
                    if (mode_ == "multi" &&
                            ((sub == 4 && en < 100. && en > 1.)
                            || ((sub != 4) && en < 1. && en > 0.02))) {
                        seedSimHit = 1;
                        break;
                    }
                }

            } // end of SimHits cycle


            // found highest-energy SimHit for single-particle
            if (mode_ != "multi" && emax_Sim > 0.) seedSimHit = 1;
        } // end of SimHits
    }// end of mc_ == "yes"

    // CYCLE OVER CELLS ========================================================
    int Ndig = 0;

    /*
    std::cout << " HcalDigiTester::reco :     nevent 1,2,3,4 = "
              << nevent1 << " " << nevent2 << " " << nevent3 << " "
              << nevent4 << std::endl;
     */

    for (digiItr = digiCollection->begin(); digiItr != digiCollection->end(); digiItr++) {

        HcalDetId cell(digiItr->id());
        int depth = cell.depth();
        //REMOVED (JRD)  int iphi = cell.iphi() - 1;
        int iphi = cell.iphi();
        int ieta = cell.ieta();
        //REMOVED (JRD)  if (ieta > 0) ieta--;
        int sub = cell.subdet();


        //  amplitude for signal cell at diff. depths
        double ampl = 0.;
        double ampl1 = 0.;
        double ampl2 = 0.;
        double ampl3 = 0.;
        double ampl4 = 0.;


        // Gains, pedestals (once !) and only for "noise" case
        if (((nevent1 == 1 && isubdet == 1) ||
                (nevent2 == 1 && isubdet == 2) ||
                (nevent3 == 1 && isubdet == 3) ||
                (nevent4 == 1 && isubdet == 4)) && noise_ == 1 && sub == isubdet) {

            HcalGenericDetId hcalGenDetId(digiItr->id());
            const HcalPedestal* pedestal = conditions->getPedestal(hcalGenDetId);
            const HcalGain* gain = conditions->getGain(hcalGenDetId);
            const HcalGainWidth* gainWidth = conditions->getGainWidth(hcalGenDetId);
            const HcalPedestalWidth* pedWidth = conditions-> getPedestalWidth(hcalGenDetId);

            for (int i = 0; i < 4; i++) {
                fill1D("HcalDigiTask_gain_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, gain->getValue(i));
                fill1D("HcalDigiTask_gainWidth_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, gainWidth->getValue(i));
                fill1D("HcalDigiTask_pedestal_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, pedestal->getValue(i));
                fill1D("HcalDigiTask_pedestal_width_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, pedWidth->getWidth(i));
            }

            fill2D("HcalDigiTask_gainMap_Depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi), gain->getValue(0));
            fill2D("HcalDigiTask_pwidthMap_Depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi), pedWidth->getWidth(0));

        }// end of event #1
        //std::cout << "==== End of event noise block in cell cycle"  << std::endl;

        if (sub == isubdet) Ndig++; // subdet number of digi

        // No-noise case, only single  subdet selected  ===========================

        if (sub == isubdet && noise_ == 0) {


            HcalCalibrations calibrations = conditions->getHcalCalibrations(cell);

            const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
	    const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
            HcalCoderDb coder(*channelCoder, *shape);
            coder.adc2fC(*digiItr, tool);

            double noiseADC = (*digiItr)[0].adc();
            double noisefC = tool[0];
            // noise evaluations from "pre-samples"
            fill1D("HcalDigiTask_ADC0_adc_depth" + str(depth) + "_" + subdet_, noiseADC);
            fill1D("HcalDigiTask_ADC0_fC_depth" + str(depth) + "_" + subdet_, noisefC);


            // OCCUPANCY maps fill
            fill2D("HcalDigiTask_ieta_iphi_occupancy_map_depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi));

            // Cycle on time slices
            // - for each Digi
            // - for one Digi with max SimHits E in subdet


            int closen = 0; // =1 if 1) seedSimHit = 1 and 2) the cell is the same
            if (ieta == ieta_Sim && iphi == iphi_Sim) closen = seedSimHit;

            for (int ii = 0; ii < tool.size(); ii++) {
                int capid = (*digiItr)[ii].capid();
                // single ts amplitude
                double val = (tool[ii] - calibrations.pedestal(capid));
/*
                if (val > 10.) {
                    if (depth == 1) strtmp = "HcalDigiTask_all_amplitudes_vs_bin_depth1_" + subdet_;
                    else strtmp = "HcalDigiTask_all_amplitudes_vs_bin_depth2_" + subdet_;
                    fill2D(strtmp, double(ii), val);
                }
*/
                if (val > 100.) {
                    if (depth == 1) strtmp = "HcalDigiTask_all_amplitudes_vs_bin_1D_depth1_" + subdet_;
                    else strtmp = "HcalDigiTask_all_amplitudes_vs_bin_1D_depth2_" + subdet_;
                    fill1D(strtmp, double(ii), val);
                }

                if (closen == 1) {
                    strtmp = "HcalDigiTask_signal_amplitude_vs_bin_all_depths_" + subdet_;
                    fill2D(strtmp, double(ii), val);
                }


                // HB/HE/HO
                if (isubdet != 4 && ii >= 4 && ii <= 7) {
                    ampl += val;
                    if (depth == 1) ampl1 += val;
                    if (depth == 2) ampl2 += val;
                    if (depth == 3) ampl3 += val;
                    if (depth == 4) ampl4 += val;

                    if (closen == 1) {
                        ampl_c += val;
                        if (depth == 1) ampl1_c += val;
                        if (depth == 2) ampl2_c += val;
                        if (depth == 3) ampl3_c += val;
                        if (depth == 4) ampl4_c += val;
                    }
                }

                // HF
                if (isubdet == 4 && ii >= 2 && ii <= 4) {
                    ampl += val;
                    if (depth == 1) ampl1 += val;
                    if (depth == 2) ampl2 += val;
                    if (depth == 3) ampl3 += val;
                    if (depth == 4) ampl4 += val;
                    if (closen == 1) {
                        ampl_c += val;
                        if (depth == 1) ampl1_c += val;
                        if (depth == 2) ampl2_c += val;
                        if (depth == 3) ampl3_c += val;
                        if (depth == 4) ampl4_c += val;

                    }
                }
            }
            // end of time bucket sample


            // maps of sum of amplitudes (sum lin.digis(4,5,6,7) - ped) all depths
/*
            strtmp = "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth1_" + subdet_;
            fill2D(strtmp, double(ieta), double(iphi), ampl1);
            strtmp = "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth2_" + subdet_;
            fill2D(strtmp, double(ieta), double(iphi), ampl2);
            strtmp = "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth3_" + subdet_;
            fill2D(strtmp, double(ieta), double(iphi), ampl3);
            strtmp = "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth4_" + subdet_;
            fill2D(strtmp, double(ieta), double(iphi), ampl4);
*/
            // just 1D of all cells' amplitudes
            strtmp = "HcalDigiTask_sum_all_amplitudes_" + subdet_;
            fill1D(strtmp, ampl);


            if (ampl1 > 10. || ampl2 > 10. || ampl3 > 10. || ampl4 > 10.) indigis++;

            // fraction 5,6 bins if ampl. is big.
            if (ampl1 > 30. && depth == 1 && closen == 1 && isubdet != 4) {
	      double fBin5 = tool[4] - calibrations.pedestal((*digiItr)[4].capid());
	      double fBin67 = tool[5] + tool[6]
		- calibrations.pedestal((*digiItr)[5].capid())
		- calibrations.pedestal((*digiItr)[6].capid());
	      
	      fBin5 /= ampl1;
	      fBin67 /= ampl1;
	      
	      strtmp = "HcalDigiTask_bin_5_frac_" + subdet_;
	      fill1D(strtmp, fBin5);
	      strtmp = "HcalDigiTask_bin_6_7_frac_" + subdet_;
	      fill1D(strtmp, fBin67);
	      
	    }
	    
	    //Special for HF
	    if (isubdet == 4 && ampl1 > 30. && depth == 1) {
	      double fBin5 = tool[2] - calibrations.pedestal((*digiItr)[2].capid());
	      double fBin67 = tool[3] + tool[4]
		- calibrations.pedestal((*digiItr)[3].capid())
		- calibrations.pedestal((*digiItr)[4].capid());
	      fBin5 /= ampl1;
	      fBin67 /= ampl1;
	      strtmp = "HcalDigiTask_bin_5_frac_" + subdet_;
	      fill1D(strtmp, fBin5);
	      strtmp = "HcalDigiTask_bin_6_7_frac_" + subdet_;
	      fill1D(strtmp, fBin67);
            }
	    
	    
            strtmp = "HcalDigiTask_signal_amplitude_" + subdet_;
            fill1D(strtmp, ampl);
            strtmp = "HcalDigiTask_signal_amplitude_depth1_" + subdet_;
            fill1D(strtmp, ampl1);
            strtmp = "HcalDigiTask_signal_amplitude_depth2_" + subdet_;
            fill1D(strtmp, ampl2);
            strtmp = "HcalDigiTask_signal_amplitude_depth3_" + subdet_;
            fill1D(strtmp, ampl3);
            strtmp = "HcalDigiTask_signal_amplitude_depth4_" + subdet_;
            fill1D(strtmp, ampl4);
        }
    } // End of CYCLE OVER CELLS =============================================

    if (isubdet != 0 && noise_ == 0) { // signal only, once per event
        strtmp = "HcalDigiTask_number_of_amplitudes_above_10fC_" + subdet_;
        fill1D(strtmp, indigis);

        // SimHits once again !!!
        double eps = 1.e-3;
        double ehits = 0.;
        double ehits1 = 0.;
        double ehits2 = 0.;
        double ehits3 = 0.;
        double ehits4 = 0.;

        if (mc_ == "yes") {
            edm::Handle<edm::PCaloHitContainer> hcalHits;
            iEvent.getByToken(tok_mc_, hcalHits);
            const edm::PCaloHitContainer * simhitResult = hcalHits.product();
            for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end(); ++simhits) {

                HcalDetId cell(simhits->id());
                int ieta = cell.ieta();
                //REMOVED (JRD)  if (ieta > 0) ieta--;
                //REMOVED (JRD)  int iphi = cell.iphi() - 1;
                int iphi = cell.iphi();
                int sub = cell.subdet();

                // take cell already found to be max energy in a particular subdet
                if (sub == isubdet && ieta == ieta_Sim && iphi == iphi_Sim) {
                    int depth = cell.depth();
                    double en = simhits->energy();

                    ehits += en;
                    if (depth == 1) ehits1 += en;
                    if (depth == 2) ehits2 += en;
                    if (depth == 3) ehits3 += en;
                    if (depth == 4) ehits4 += en;
                }
            }

            strtmp = "HcalDigiTask_amplitude_vs_simhits_" + subdet_;
            if (ehits > eps) fill2D(strtmp, ehits, ampl_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_depth1_" + subdet_;
            if (ehits1 > eps) fill2D(strtmp, ehits1, ampl1_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_depth2_" + subdet_;
            if (ehits2 > eps) fill2D(strtmp, ehits2, ampl2_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_depth3_" + subdet_;
            if (ehits3 > eps) fill2D(strtmp, ehits3, ampl3_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_depth4_" + subdet_;
            if (ehits4 > eps) fill2D(strtmp, ehits4, ampl4_c);

            strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_" + subdet_;
            if (ehits > eps) fillPf(strtmp, ehits, ampl_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth1_" + subdet_;
            if (ehits1 > eps) fillPf(strtmp, ehits1, ampl1_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth2_" + subdet_;
            if (ehits2 > eps) fillPf(strtmp, ehits2, ampl2_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth3_" + subdet_;
            if (ehits3 > eps) fillPf(strtmp, ehits3, ampl3_c);
            strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth4_" + subdet_;
            if (ehits4 > eps) fillPf(strtmp, ehits4, ampl4_c);

            strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_" + subdet_;
            if (ehits > eps) fill1D(strtmp, ampl_c / ehits);
            strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_depth1_" + subdet_;
            if (ehits1 > eps) fill1D(strtmp, ampl1_c / ehits1);
            strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_depth2_" + subdet_;
            if (ehits2 > eps) fill1D(strtmp, ampl2_c / ehits2);
            strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_depth3_" + subdet_;
            if (ehits3 > eps) fill1D(strtmp, ampl3_c / ehits3);
            strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_depth4_" + subdet_;
            if (ehits4 > eps) fill1D(strtmp, ampl4_c / ehits4);

        } // end of if(mc_ == "yes")

        strtmp = "HcalDigiTask_Ndigis_" + subdet_;
        fill1D(strtmp, double(Ndig));

    } //  end of if( subdet != 0 && noise_ == 0) { // signal only
}

void HcalDigisValidation::eval_occupancy() {

    std::string strtmp;

    float fev = float (nevtot);
        // std::cout << "*** nevtot " <<  nevtot << std::endl;

    float sumphi_1, sumphi_2, sumphi_3, sumphi_4;
    float phi_factor;
    float cnorm;

    strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth1_" + subdet_;
    int nx = monitor(strtmp)->getNbinsX();
    int ny = monitor(strtmp)->getNbinsY();

    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {

            // occupancies

            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth1_" + subdet_;
            cnorm = monitor(strtmp)->getBinContent(i, j) / fev;
            monitor(strtmp)->setBinContent(i, j, cnorm);

            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth2_" + subdet_;
            cnorm = monitor(strtmp)->getBinContent(i, j) / fev;
            monitor(strtmp)->setBinContent(i, j, cnorm);

            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth3_" + subdet_;
            cnorm = monitor(strtmp)->getBinContent(i, j) / fev;
            monitor(strtmp)->setBinContent(i, j, cnorm);

            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth4_" + subdet_;
            cnorm = monitor(strtmp)->getBinContent(i, j) / fev;
            monitor(strtmp)->setBinContent(i, j, cnorm);

        }
    }	    

    for (int i = 1; i <= 82; i++) {

        int ieta = i - 42; // -41 -1, 0 40
        if (ieta >= 0) ieta += 1; // -41 -1, 1 41  - to make it detector-like

        if (ieta >= -20 && ieta <= 20) {
            phi_factor = 72.;
        } else {
            if (ieta >= 40 || ieta <= -40) {
                phi_factor = 18.;
            } else
                phi_factor = 36.;
        }

        sumphi_1 = 0.;
        sumphi_2 = 0.;
        sumphi_3 = 0.;
        sumphi_4 = 0.;

        for (int iphi = 1; iphi <= 72; iphi++) {
            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth1_" + subdet_;
            sumphi_1 += monitor(strtmp)->getTH1()->GetBinContent(monitor(strtmp)->getTH1()->FindFixBin(double(ieta),double(iphi)));
            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth2_" + subdet_;
            sumphi_2 += monitor(strtmp)->getTH1()->GetBinContent(monitor(strtmp)->getTH1()->FindFixBin(double(ieta),double(iphi)));
            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth3_" + subdet_;
            sumphi_3 += monitor(strtmp)->getTH1()->GetBinContent(monitor(strtmp)->getTH1()->FindFixBin(double(ieta),double(iphi)));
            strtmp = "HcalDigiTask_ieta_iphi_occupancy_map_depth4_" + subdet_;
            sumphi_4 += monitor(strtmp)->getTH1()->GetBinContent(monitor(strtmp)->getTH1()->FindFixBin(double(ieta),double(iphi)));
        }

        //REMOVED (JRD) if (ieta >= 0) ieta -= 1; // -41 -1, 0 40  - to bring back to strtmp num !!!
        double deta = double(ieta);

        // occupancies vs ieta
        cnorm = sumphi_1 / phi_factor;
        strtmp = "HcalDigiTask_occupancy_vs_ieta_depth1_" + subdet_;
        fill1D(strtmp, deta, cnorm);

        cnorm = sumphi_2 / phi_factor;
        strtmp = "HcalDigiTask_occupancy_vs_ieta_depth2_" + subdet_;
        fill1D(strtmp, deta, cnorm);

        cnorm = sumphi_3 / phi_factor;
        strtmp = "HcalDigiTask_occupancy_vs_ieta_depth3_" + subdet_;
        fill1D(strtmp, deta, cnorm);

        cnorm = sumphi_4 / phi_factor;
        strtmp = "HcalDigiTask_occupancy_vs_ieta_depth4_" + subdet_;
        fill1D(strtmp, deta, cnorm);

    } // end of i-loop

}

void HcalDigisValidation::book1D(DQMStore::IBooker &ib, std::string name, int n, double min, double max) {
    if (!msm_->count(name)) (*msm_)[name] = ib.book1D(name.c_str(), name.c_str(), n, min, max);
}

void HcalDigisValidation::book1D(DQMStore::IBooker &ib, std::string name, const HistLim& limX) {
    if (!msm_->count(name)) (*msm_)[name] = ib.book1D(name.c_str(), name.c_str(), limX.n, limX.min, limX.max);
}

void HcalDigisValidation::fill1D(std::string name, double X, double weight) {
    msm_->find(name)->second->Fill(X, weight);
}

void HcalDigisValidation::book2D(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY) {
    if (!msm_->count(name)) (*msm_)[name] = ib.book2D(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
}

void HcalDigisValidation::fill2D(std::string name, double X, double Y, double weight) {
    msm_->find(name)->second->Fill(X, Y, weight);
}

void HcalDigisValidation::bookPf(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY) {
    if (!msm_->count(name)) (*msm_)[name] = ib.bookProfile(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
}

void HcalDigisValidation::bookPf(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY, const char *option) {
    if (!msm_->count(name)) (*msm_)[name] = ib.bookProfile(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max, option);
}

void HcalDigisValidation::fillPf(std::string name, double X, double Y) {
    msm_->find(name)->second->Fill(X, Y);
}

MonitorElement* HcalDigisValidation::monitor(std::string name) {
    if (!msm_->count(name)) return NULL;
    else return msm_->find(name)->second;
}

std::string HcalDigisValidation::str(int x) {
    std::stringstream out;
    out << x;
    return out.str();
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigisValidation);


