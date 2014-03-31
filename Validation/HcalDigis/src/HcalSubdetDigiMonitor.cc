#include "Validation/HcalDigis/src/HcalSubdetDigiMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

struct HistLim
{
  HistLim(int nbin, float mini, float maxi)
  : n(nbin), min(mini), max(maxi) {}
  int n;
  float min;
  float max;
};

HcalSubdetDigiMonitor::HcalSubdetDigiMonitor(DQMStore* dbe, 
                                             const std::string & subdet, int noise)
: dbe_(dbe),
  subdet_(subdet),
  noise_(noise)
{

  // defaults are for HB

  HistLim Ndigis(2600,0.,2600.);
  HistLim ndigis(50, 0., 50.);
  HistLim sime(200, 0., 1.0);
  HistLim digiAmp(700, -100., 600.);
  HistLim ratio(250, 0., 2500.);
  HistLim sumAmp(100, -500., 1500.);

  HistLim nbin(10,0.,10.);

  HistLim pedestal(75, 0., 15.);
  HistLim pedestalfC(400,-10.,30.);

  HistLim frac(52, -0.02, 1.02);

  HistLim pedLim(80, 0., 8.); 
  HistLim pedWidthLim(100, 0., 2.); 

  HistLim gainLim(120, 0.,0.6); 
  HistLim gainWidthLim(100, 0.,0.3); 

  HistLim ietaLim( 82, -41., 41.);
  HistLim iphiLim(72, 0., 72.);

  if(subdet_ == "HE")
    {
      sime        = HistLim(200, 0., 1.0);
      digiAmp     = HistLim(250, -100., 400.);
    }
  else if(subdet_ == "HF")
    {
      ndigis      = HistLim(500, 0., 500.);
      sime        = HistLim(100, 0., 100.);
      digiAmp     = HistLim(420, -100., 2000.);
      ratio       = HistLim(120, 0., 120.);
      pedLim      = HistLim(100, 0., 20.); 
      pedWidthLim = HistLim(100, 0., 5.); 

    }
  else if(subdet_ == "HO")
    {
      sime    = HistLim(200, 0., 1.0);
      digiAmp = HistLim(200, 0., 200.);
      gainLim = HistLim(150, 0., 1.5); 
   }
  
  Char_t histo[100];
  const char * sub = subdet_.c_str();
  if ( dbe_ ) {
    dbe_->setCurrentFolder("HcalDigisV/HcalDigiTask");
  }

  /*
  std::cout << " HcalSubdetDigiMonitor : "
	    << "  subdet " << subdet_ 
	    << "  noise_ " << noise_ 
	    << std::endl;
  */

  if(noise_ == 0) {   

    // number of digis in each subdetector
        
    sprintf (histo, "HcalDigiTask_Ndigis_%s", sub ) ;
    meNdigis = book1D(histo, Ndigis);

    // maps of occupancies
    sprintf (histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth1_%s", sub ) ;
    meOccupancy_map_depth1 = book2D(histo,  ietaLim, iphiLim);

    sprintf (histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth2_%s", sub ) ;
    meOccupancy_map_depth2 = book2D(histo,  ietaLim, iphiLim);

    sprintf (histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth3_%s", sub ) ;
    meOccupancy_map_depth3 = book2D(histo,  ietaLim, iphiLim);

    sprintf (histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth4_%s", sub ) ;
    meOccupancy_map_depth4 = book2D(histo,  ietaLim, iphiLim);

    // occupancies vs ieta
    sprintf (histo, "HcalDigiTask_occupancy_vs_ieta_depth1_%s", sub ) ;
    meOccupancy_vs_ieta_depth1 = book1D(histo,  ietaLim);

    sprintf (histo, "HcalDigiTask_occupancy_vs_ieta_depth2_%s", sub ) ;
    meOccupancy_vs_ieta_depth2 = book1D(histo,  ietaLim);

    sprintf (histo, "HcalDigiTask_occupancy_vs_ieta_depth3_%s", sub ) ;
    meOccupancy_vs_ieta_depth3 = book1D(histo,  ietaLim);

    sprintf (histo, "HcalDigiTask_occupancy_vs_ieta_depth4_%s", sub ) ;
    meOccupancy_vs_ieta_depth4 = book1D(histo,  ietaLim);
  

    // maps of sum of amplitudes (sum lin.digis(4,5,6,7) - ped) all depths
    sprintf (histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth1_%s", sub ) ;
    meAmplIetaIphi1= book2D(histo,  ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth2_%s", sub ) ;
    meAmplIetaIphi2= book2D(histo,  ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth3_%s", sub ) ;
    meAmplIetaIphi3= book2D(histo,  ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_depth4_%s", sub ) ;
    meAmplIetaIphi4= book2D(histo,  ietaLim, iphiLim);
    // just 1D of all cells' amplitudes 
    sprintf (histo, "HcalDigiTask_sum_all_amplitudes_%s", sub);
    meSumAmp = book1D(histo, sumAmp);

    sprintf (histo, "HcalDigiTask_number_of_amplitudes_above_10fC_%s", sub);
    menDigis = book1D(histo, ndigis);

    sprintf (histo, "HcalDigiTask_ADC0_adc_depth1_%s", sub);
    meADC0_depth1 = book1D(histo, pedestal);
    sprintf (histo, "HcalDigiTask_ADC0_adc_depth2_%s", sub);
    meADC0_depth2 = book1D(histo, pedestal);
    sprintf (histo, "HcalDigiTask_ADC0_adc_depth3_%s", sub);
    meADC0_depth3 = book1D(histo, pedestal);
    sprintf (histo, "HcalDigiTask_ADC0_adc_depth4_%s", sub);
    meADC0_depth4 = book1D(histo, pedestal);
   
    sprintf (histo, "HcalDigiTask_ADC0_fC_depth1_%s", sub);
    meADC0fC_depth1 = book1D(histo, pedestalfC);    
    sprintf (histo, "HcalDigiTask_ADC0_fC_depth2_%s", sub);
    meADC0fC_depth2 = book1D(histo, pedestalfC);
    sprintf (histo, "HcalDigiTask_ADC0_fC_depth3_%s", sub);
    meADC0fC_depth3 = book1D(histo, pedestalfC);
    sprintf (histo, "HcalDigiTask_ADC0_fC_depth4_%s", sub);
    meADC0fC_depth4 = book1D(histo, pedestalfC);

    sprintf (histo, "HcalDigiTask_signal_amplitude_%s", sub);
    meSignalAmp  = book1D(histo, digiAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth1_%s", sub);
    meSignalAmp1 = book1D(histo, digiAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth2_%s", sub);
    meSignalAmp2 = book1D(histo, digiAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth3_%s", sub);
    meSignalAmp3 = book1D(histo, digiAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth4_%s", sub);
    meSignalAmp4 = book1D(histo, digiAmp );

  
    sprintf (histo, "HcalDigiTask_signal_amplitude_vs_bin_all_depths_%s", sub);
    meSignalTimeSlice = book2D(histo, nbin, digiAmp);

    sprintf (histo, "HcalDigiTask_all_amplitudes_vs_bin_depth1%s", sub);
    meAll10slices_depth1 = book2D(histo, nbin, digiAmp);
    sprintf (histo, "HcalDigiTask_all_amplitudes_vs_bin_depth2%s", sub);
    meAll10slices_depth2 = book2D(histo, nbin, digiAmp);

    sprintf (histo, "HcalDigiTask_all_amplitudes_vs_bin_1D_depth1_%s", sub);
    meAll10slices1D_depth1 = book1D(histo, nbin);
    sprintf (histo, "HcalDigiTask_all_amplitudes_vs_bin_1D_depth2_%s", sub);
    meAll10slices1D_depth2 = book1D(histo, nbin);

    sprintf (histo, "HcalDigiTask_bin_5_frac_%s", sub);
    meBin5Frac = book1D(histo, frac);
    sprintf (histo, "HcalDigiTask_bin_6_7_frac_%s", sub);
    meBin67Frac = book1D(histo, frac);

    
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_%s", sub);
    meDigiSimhit = book2D(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_depth1_%s", sub);
    meDigiSimhit1 = book2D(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_depth2_%s", sub);
    meDigiSimhit2 = book2D(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_depth3_%s", sub);
    meDigiSimhit3 = book2D(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_depth4_%s", sub);
    meDigiSimhit4 = book2D(histo, sime, digiAmp);
  
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_profile_%s", sub);
    meDigiSimhitProfile = bookProfile(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth1_%s",sub);
    meDigiSimhitProfile1 = bookProfile(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth2_%s",sub);
    meDigiSimhitProfile2 = bookProfile(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth3_%s",sub);
    meDigiSimhitProfile3 = bookProfile(histo, sime, digiAmp);
    sprintf (histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth4_%s",sub);
    meDigiSimhitProfile4 = bookProfile(histo, sime, digiAmp);
  
    sprintf (histo, "HcalDigiTask_ratio_amplitude_vs_simhits_%s", sub);
    meRatioDigiSimhit = book1D(histo, ratio);
    sprintf (histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth1_%s", sub);
    meRatioDigiSimhit1 = book1D(histo, ratio);
    sprintf (histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth2_%s", sub);
    meRatioDigiSimhit2 = book1D(histo, ratio);
    sprintf (histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth3_%s", sub);
    meRatioDigiSimhit3 = book1D(histo, ratio);
    sprintf (histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth4_%s", sub);
    meRatioDigiSimhit4 = book1D(histo, ratio);

  }
  else {  // noise only  
    
    // EVENT "1" distributions of all cells properties 
    
         
    if(subdet_ == "HB" || subdet_ == "HE" || subdet_ == "HF") {
      sprintf (histo, "HcalDigiTask_gain_capId0_Depth1_%s", sub);
      meGain0Depth1 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId1_Depth1_%s", sub);
      meGain1Depth1 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId2_Depth1_%s", sub);
      meGain2Depth1 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId3_Depth1_%s", sub);
      meGain3Depth1 = book1D(histo,gainLim);

      sprintf (histo, "HcalDigiTask_gain_capId0_Depth2_%s", sub);
      meGain0Depth2 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId1_Depth2_%s", sub);
      meGain1Depth2 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId2_Depth2_%s", sub);
      meGain2Depth2 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId3_Depth2_%s", sub);
      meGain3Depth2 = book1D(histo,gainLim);

      sprintf (histo, "HcalDigiTask_gainWidth_capId0_Depth1_%s", sub);
      meGainWidth0Depth1 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId1_Depth1_%s", sub);
      meGainWidth1Depth1 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId2_Depth1_%s", sub);
      meGainWidth2Depth1 = book1D(histo,gainWidthLim);    
      sprintf (histo, "HcalDigiTask_gainWidth_capId3_Depth1_%s", sub);
      meGainWidth3Depth1 = book1D(histo,gainWidthLim);
    
      sprintf (histo, "HcalDigiTask_gainWidth_capId0_Depth2_%s", sub);
      meGainWidth0Depth2 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId1_Depth2_%s", sub);
      meGainWidth1Depth2 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId2_Depth2_%s", sub);
      meGainWidth2Depth2 = book1D(histo,gainWidthLim);    
      sprintf (histo, "HcalDigiTask_gainWidth_capId3_Depth2_%s", sub);
      meGainWidth3Depth2 = book1D(histo,gainWidthLim);
    
      sprintf (histo, "HcalDigiTask_pedestal_capId0_Depth1_%s", sub);
      mePed0Depth1 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId1_Depth1_%s", sub);
      mePed1Depth1 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId2_Depth1_%s", sub);
      mePed2Depth1 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId3_Depth1_%s", sub);
      mePed3Depth1 = book1D(histo,pedLim);

      sprintf (histo, "HcalDigiTask_pedestal_capId0_Depth2_%s", sub);
      mePed0Depth2 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId1_Depth2_%s", sub);
      mePed1Depth2 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId2_Depth2_%s", sub);
      mePed2Depth2 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId3_Depth2_%s", sub);
      mePed3Depth2 = book1D(histo,pedLim);

      sprintf (histo, "HcalDigiTask_pedestal_width_capId0_Depth1_%s", sub);
      mePedWidth0Depth1 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId1_Depth1_%s", sub);
      mePedWidth1Depth1 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId2_Depth1_%s", sub);
      mePedWidth2Depth1 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId3_Depth1_%s", sub);
      mePedWidth3Depth1 = book1D(histo,pedWidthLim);

      sprintf (histo, "HcalDigiTask_pedestal_width_capId0_Depth2_%s", sub);
      mePedWidth0Depth2 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId1_Depth2_%s", sub);
      mePedWidth1Depth2 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId2_Depth2_%s", sub);
      mePedWidth2Depth2 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId3_Depth2_%s", sub);
      mePedWidth3Depth2 = book1D(histo,pedWidthLim);


    }

    if(subdet_ == "HE") {
      sprintf (histo, "HcalDigiTask_gain_capId0_Depth3_%s", sub);
      meGain0Depth3 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId1_Depth3_%s", sub);
      meGain1Depth3 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId2_Depth3_%s", sub);
      meGain2Depth3 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId3_Depth3_%s", sub);
      meGain3Depth3 = book1D(histo,gainLim);

      sprintf (histo, "HcalDigiTask_gainWidth_capId0_Depth3_%s", sub);
      meGainWidth0Depth3 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId1_Depth3_%s", sub);
      meGainWidth1Depth3 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId2_Depth3_%s", sub);
      meGainWidth2Depth3 = book1D(histo,gainWidthLim);    
      sprintf (histo, "HcalDigiTask_gainWidth_capId3_Depth3_%s", sub);
      meGainWidth3Depth3 = book1D(histo,gainWidthLim);
    
      sprintf (histo, "HcalDigiTask_pedestal_capId0_Depth3_%s", sub);
      mePed0Depth3 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId1_Depth3_%s", sub);
      mePed1Depth3 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId2_Depth3_%s", sub);
      mePed2Depth3 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId3_Depth3_%s", sub);
      mePed3Depth3 = book1D(histo,pedLim);

      sprintf (histo, "HcalDigiTask_pedestal_width_capId0_Depth3_%s", sub);
      mePedWidth0Depth3 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId1_Depth3_%s", sub);
      mePedWidth1Depth3 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId2_Depth3_%s", sub);
      mePedWidth2Depth3 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId3_Depth3_%s", sub);
      mePedWidth3Depth3 = book1D(histo,pedWidthLim);

    }

    if(subdet_ == "HO") {
      sprintf (histo, "HcalDigiTask_gain_capId0_Depth4_%s", sub);
      meGain0Depth4 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId1_Depth4_%s", sub);
      meGain1Depth4 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId2_Depth4_%s", sub);
      meGain2Depth4 = book1D(histo,gainLim);
      sprintf (histo, "HcalDigiTask_gain_capId3_Depth4_%s", sub);
      meGain3Depth4 = book1D(histo,gainLim);

      sprintf (histo, "HcalDigiTask_gainWidth_capId0_Depth4_%s", sub);
      meGainWidth0Depth4 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId1_Depth4_%s", sub);
      meGainWidth1Depth4 = book1D(histo,gainWidthLim);
      sprintf (histo, "HcalDigiTask_gainWidth_capId2_Depth4_%s", sub);
      meGainWidth2Depth4 = book1D(histo,gainWidthLim);    
      sprintf (histo, "HcalDigiTask_gainWidth_capId3_Depth4_%s", sub);
      meGainWidth3Depth4 = book1D(histo,gainWidthLim);

    
      sprintf (histo, "HcalDigiTask_pedestal_capId0_Depth4_%s", sub);
      mePed0Depth4 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId1_Depth4_%s", sub);
      mePed1Depth4 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId2_Depth4_%s", sub);
      mePed2Depth4 = book1D(histo,pedLim);
      sprintf (histo, "HcalDigiTask_pedestal_capId3_Depth4_%s", sub);
      mePed3Depth4 = book1D(histo,pedLim);

      sprintf (histo, "HcalDigiTask_pedestal_width_capId0_Depth4_%s", sub);
      mePedWidth0Depth4 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId1_Depth4_%s", sub);
      mePedWidth1Depth4 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId2_Depth4_%s", sub);
      mePedWidth2Depth4 = book1D(histo,pedWidthLim);
      sprintf (histo, "HcalDigiTask_pedestal_width_capId3_Depth4_%s", sub);
      mePedWidth3Depth4 = book1D(histo,pedWidthLim);

    }

    sprintf (histo, "HcalDigiTask_gainMap_Depth1_%s", sub);
    meGainMap1 = book2D(histo, ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_gainMap_Depth2_%s", sub);
    meGainMap2 = book2D(histo, ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_gainMap_Depth3_%s", sub);
    meGainMap3 = book2D(histo, ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_gainMap_Depth4_%s", sub);
    meGainMap4 = book2D(histo, ietaLim, iphiLim);
    
    sprintf (histo, "HcalDigiTask_pwidthMap_Depth1_%s", sub);
    mePwidthMap1 = book2D(histo, ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_pwidthMap_Depth2_%s", sub);
    mePwidthMap2 = book2D(histo, ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_pwidthMap_Depth3_%s", sub);
    mePwidthMap3 = book2D(histo, ietaLim, iphiLim);
    sprintf (histo, "HcalDigiTask_pwidthMap_Depth4_%s", sub);
    mePwidthMap4 = book2D(histo, ietaLim, iphiLim);
 
  } //end of noise-only
}


MonitorElement * HcalSubdetDigiMonitor::book1D(const std::string & name, 
                                    HcalSubdetDigiMonitor::HistLim lim)
{
  return dbe_->book1D(name, name, lim.n, lim.min, lim.max);
}


MonitorElement * HcalSubdetDigiMonitor::book2D(const std::string & name,
                                    HcalSubdetDigiMonitor::HistLim lim1,
                                    HcalSubdetDigiMonitor::HistLim lim2)
{
  return dbe_->book2D(name, name, lim1.n, lim1.min, lim1.max,
                                  lim2.n, lim2.min, lim2.max);
}

MonitorElement * HcalSubdetDigiMonitor::bookProfile(const std::string & name,
                                    HcalSubdetDigiMonitor::HistLim lim1,
                                    HcalSubdetDigiMonitor::HistLim lim2)
{
  return dbe_->bookProfile(name, name, lim1.n, lim1.min, lim1.max,
                                  lim2.n, lim2.min, lim2.max);
}



