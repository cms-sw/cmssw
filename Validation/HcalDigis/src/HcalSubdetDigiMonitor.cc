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

  HistLim ndigis(50, 0., 50.);
  HistLim sime(150, 0., 1.5);
  HistLim digiAmp(650,-100., 1200.);
  HistLim ratio(200, 0., 2000.);
  HistLim sumAmp(100, -500., 1500.);

  HistLim signalAmp(151, -10., 1500.);  

  HistLim nbin(10,0.,10.);
  HistLim signal(160,-100., 1500.);

  HistLim pedestal(75, 0., 15.);
  HistLim pedestalfC(400,-10.,30.);

  HistLim frac(52, -0.02, 1.02);

  HistLim pedLim(200, 0., 20.); 
  HistLim pedWidthLim(100, 0., 2.); 

  HistLim gainLim(100, 0.,0.3); 
  HistLim gainWidthLim(100, 0.,0.3); 

  HistLim ietaLim( 82, -41., 41.);
  HistLim iphiLim(72, 0., 72.);

  if(subdet_ == "HE")
    {
      sime    = HistLim(100, 0., 1.);
    }
  else if(subdet_ == "HF")
    {
      ndigis    = HistLim(20, 0., 20.);
      digiAmp   = HistLim(550, -100., 1000.);
      ratio     = HistLim(50, 0., 50.);
      sime      = HistLim(40, 0., 40.);
      signalAmp = HistLim(101, -10., 1000.);  
    }
  else if(subdet_ == "HO")
    {
      digiAmp = HistLim(300, 0., 150.);
      sime    = HistLim(100, 0., 0.2);
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


    // map of sum of amplitudes (sum lin.digis(4,5,6,7) - ped) all depths
    sprintf (histo, "HcalDigiTask_ieta_iphi_map_of_amplitudes_fC_%s", sub ) ;
    meAmplIetaIphi= book2D(histo,  ietaLim, iphiLim);
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
    meSignalAmp = book1D(histo, signalAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth1_%s", sub);
    meSignalAmp1 = book1D(histo, signalAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth2_%s", sub);
    meSignalAmp2 = book1D(histo, signalAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth3_%s", sub);
    meSignalAmp3 = book1D(histo, signalAmp );
    sprintf (histo, "HcalDigiTask_signal_amplitude_depth4_%s", sub);
    meSignalAmp4 = book1D(histo, signalAmp );

  
    sprintf (histo, "HcalDigiTask_signal_amplitude_vs_bin_all_depths_%s", sub);
    meSignalTimeSlice = book2D(histo, nbin, signal);

    sprintf (histo, "HcalDigiTask_all_amplitudes_vs_bin_%s", sub);
    meAll10slices = book2D(histo, nbin, signal);

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
    
    sprintf (histo, "HcalDigiTask_pedestal_capId0_%s", sub);
    mePedCapId0 = book1D(histo,pedLim);
    sprintf (histo, "HcalDigiTask_pedestal_capId1_%s", sub);
    mePedCapId1 = book1D(histo,pedLim);
    sprintf (histo, "HcalDigiTask_pedestal_capId2_%s", sub);
    mePedCapId2 = book1D(histo,pedLim);
    sprintf (histo, "HcalDigiTask_pedestal_capId3_%s", sub);
    mePedCapId3 = book1D(histo,pedLim);
    
    sprintf (histo, "HcalDigiTask_pedestal_width_capId0_%s", sub);
    mePedWidthCapId0 = book1D(histo,pedWidthLim);
    sprintf (histo, "HcalDigiTask_pedestal_width_capId1_%s", sub);
    mePedWidthCapId1 = book1D(histo,pedWidthLim);
    sprintf (histo, "HcalDigiTask_pedestal_width_capId2_%s", sub);
    mePedWidthCapId2 = book1D(histo,pedWidthLim);
    sprintf (histo, "HcalDigiTask_pedestal_width_capId3_%s", sub);
    mePedWidthCapId3 = book1D(histo,pedWidthLim);
     
    sprintf (histo, "HcalDigiTask_gain_Depth1_%s", sub);
    meGainDepth1 = book1D(histo,gainLim);
    sprintf (histo, "HcalDigiTask_gain_Depth2_%s", sub);
    meGainDepth2 = book1D(histo,gainLim);
    sprintf (histo, "HcalDigiTask_gain_Depth3_%s", sub);
    meGainDepth3 = book1D(histo,gainLim);
    sprintf (histo, "HcalDigiTask_gain_Depth4_%s", sub);
    meGainDepth4 = book1D(histo,gainLim);
    
    sprintf (histo, "HcalDigiTask_gainWidth_Depth1_%s", sub);
    meGainWidthDepth1 = book1D(histo,gainWidthLim);
    sprintf (histo, "HcalDigiTask_gainWidth_Depth2_%s", sub);
    meGainWidthDepth2 = book1D(histo,gainWidthLim);
    sprintf (histo, "HcalDigiTask_gainWidth_Depth3_%s", sub);
    meGainWidthDepth3 = book1D(histo,gainWidthLim);
    sprintf (histo, "HcalDigiTask_gainWidth_Depth4_%s", sub);
    meGainWidthDepth4 = book1D(histo,gainWidthLim);
    
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



