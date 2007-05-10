#include "Validation/HcalDigis/src/HcalSubdetDigiMonitor.h"

struct HistLim
{
  HistLim(int nbin, float mini, float maxi)
  : n(nbin), min(mini), max(maxi) {}
  int n;
  float min;
  float max;
};

HcalSubdetDigiMonitor::HcalSubdetDigiMonitor(DaqMonitorBEInterface* dbe, 
                                             const std::string & subdet)
: dbe_(dbe),
  subdet_(subdet),
  meEta(0),
  mePhi(0),
  meDigiSimhit(0),
  meRatioDigiSimhit(0),
  meDigiSimhitProfile(0),
  menDigis(0),
  meSumDigis(0),
  meADC0(0),
  meBin5Frac(0),
  meBin67Frac(0),
  meBin4567Frac(0),
  mePedestalfC(0),
  meEtaMC(0),
  mePhiMC(0),
  meNTowersGt10(0),
  meTimeSlice(0)
{
  // for Time Slice limit
  HistLim nbin(10,0.,10.);
  HistLim signalPlot(1100,-100., 1000.);
    
  HistLim NTowersGt10(40, 0., 40.);

  // defaults are for HB
  HistLim etaPlot(40, -1.74, 1.74);
  HistLim phiPlot(72, -3.14159, 3.14159);
  HistLim ndigisPlot(40, 0., 80.);
  HistLim simePlot(50, 0., 1.5);
  HistLim digiAmpPlot(30000,-1000., 1200.);
  HistLim ratioPlot(50, 0., 2000.);
  float sumWithNoiseMax = 500.;
  HistLim pedestalPlot(3000, 0., 30.);
  HistLim fracPlot(52, -0.02, 1.02);
  HistLim fracPlot4567(50, 0. , 1000.);
  HistLim pedestalfCPlot(3000,-30.,30.);
  HistLim etaMCPlot(1000, -5., 5.);
  HistLim phiMCPlot(1000, -3.15, 3.15); 
  if(subdet_ == "HE")
  {
    etaPlot = HistLim(60, -3., 3.);
    phiPlot.n = 36;
    ndigisPlot.max = 250.;
    simePlot.max = 1.;
    digiAmpPlot=HistLim(10000, -1000., 1200.);
  }
  else if(subdet_ == "HF")
  {
    etaPlot = HistLim(100, -5., 5.);
    phiPlot.n = 36;
    ndigisPlot = HistLim(20, 0., 20.);
    digiAmpPlot = HistLim(11000, -100., 1000.);
    ratioPlot = HistLim(50, 0., 50.);
    sumWithNoiseMax = 150.;
    simePlot= HistLim(30, 0., 60.);
  }
  else if(subdet_ == "HO")
  {
    digiAmpPlot = HistLim(1500, 0., 150.);
    ratioPlot.max = 1400.;
    ndigisPlot = HistLim(50, 0., 50.);
    simePlot.max = 0.2;
  }

  Char_t histo[100];
  const char * sub = subdet_.c_str();
  dbe_->setCurrentFolder("HcalDigiTask");
  sprintf (histo, "HcalDigiTask_eta_of_digis_%s", sub ) ;
  meEta= book1D(histo, etaPlot);
  sprintf (histo, "HcalDigiTask_phi_of_digis_1gev_%s", sub ) ;
  mePhi= book1D(histo, phiPlot);
  sprintf (histo, "HcalDigiTask_energy_digis4567_vs_simhits_%s", sub);
  meDigiSimhit= book2D(histo, simePlot, digiAmpPlot);
  sprintf (histo, "HcalDigiTask_ratio_energy_digis4567_vs_simhits_%s", sub);
  meRatioDigiSimhit= book1D(histo, ratioPlot);

  sprintf (histo, "HcalDigiTask_energy_digis4567_vs_simhits_profile_%s", sub);
  meDigiSimhitProfile = bookProfile(histo, simePlot, digiAmpPlot);

  sprintf (histo, "HcalDigiTask_number_of_digis_%s", sub);
  menDigis= book1D(histo, ndigisPlot);

  sprintf (histo, "HcalDigiTask_sum_of_digis_fC_%s", sub);
  meSumDigis= book1D(histo, HistLim(50, 0., 1200.));

  sprintf (histo, "HcalDigiTask_signal_vs_bin_%s", sub);
  meTimeSlice = book2D(histo, nbin, signalPlot);


  sprintf (histo, "HcalDigiTask_ADC0_adc_count_%s", sub);
  meADC0= book1D(histo, pedestalPlot);
  sprintf (histo, "HcalDigiTask_bin_5_frac_%s", sub);
  meBin5Frac = book1D(histo, fracPlot);
  sprintf (histo, "HcalDigiTask_bin_6_7_frac_%s", sub);
  meBin67Frac = book1D(histo, fracPlot);

  sprintf (histo, "HcalDigiTask_sum_bin_4567_gt_10_fC_%s", sub);
  meBin4567Frac = book1D(histo, fracPlot4567);



  sprintf (histo, "HcalDigiTask_subtracted_pedestal_fC_%s", sub);
  mePedestalfC = book1D(histo,pedestalfCPlot);

  sprintf (histo, "HcalDigiTask_ADC0_fC_%s", sub);
  meDigifC = book1D(histo,pedestalfCPlot);

  sprintf (histo, "HcalDigiTask_linerized_digis_minus_pedestal_fC_%s", sub);
  meDigiMinusPedfC = book1D(histo,pedestalfCPlot);


 sprintf (histo, "HcalDigiTask_phi_mc_particle_%s", sub);
 mePhiMC = book1D(histo,phiMCPlot);
 sprintf (histo, "HcalDigiTask_eta_mc_particle_%s", sub);
 meEtaMC = book1D(histo,etaMCPlot);


 sprintf (histo, "HcalDigiTask_number_of_tower_gt_10_fC_%s", sub);
  meNTowersGt10= book1D(histo,NTowersGt10);

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



