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
  meSumDigis_noise(0),
  mePedestal(0),
  meBin5Frac(0),
  meBin67Frac(0)
{

  // defaults are for HB
  HistLim etaPlot(40, -1.74, 1.74);
  HistLim phiPlot(72, -3.14159, 3.14159);
  HistLim ndigisPlot(40, 0., 80.);
  HistLim simePlot(50, 0., 1.5);
  HistLim digiAmpPlot(8000, 0., 800.);
  HistLim ratioPlot(200, 0., 1000.);
  float sumWithNoiseMax = 500.;
  HistLim pedestalPlot(50, 0., 20.);
  HistLim fracPlot(52, -0.02, 1.02);

  if(subdet_ == "HE")
  {
    etaPlot = HistLim(60, -3., 3.);
    phiPlot.n = 36;
    ndigisPlot.max = 250.;
    simePlot.max = 1.;
    ratioPlot.max = 1400.;
    digiAmpPlot=HistLim(10000, 0., 1000.);
  }
  else if(subdet_ == "HF")
  {
    etaPlot = HistLim(100, -5., 5.);
    phiPlot.n = 36;
    ndigisPlot = HistLim(20, 0., 20.);
    digiAmpPlot = HistLim(3500, 0., 350.);
    ratioPlot = HistLim(200, 3., 7.);
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
  sprintf (histo, "HcalDigiTask_phi_of_digis_%s", sub ) ;
  mePhi= book1D(histo, phiPlot);
  sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_%s", sub);
  meDigiSimhit= book2D(histo, simePlot, digiAmpPlot);
  sprintf (histo, "HcalDigiTask_ratio_energy_digis_vs_simhits_%s", sub);
  meRatioDigiSimhit= book1D(histo, ratioPlot);

  sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_profile_%s", sub);
  meDigiSimhitProfile = bookProfile(histo, simePlot, digiAmpPlot);

  sprintf (histo, "HcalDigiTask_number_of_digis_%s", sub);
  menDigis= book1D(histo, ndigisPlot);

  sprintf (histo, "HcalDigiTask_sum_over_digis_fC_%s", sub);
  meSumDigis= book1D(histo, HistLim(50, 0., digiAmpPlot.max));

  sprintf (histo, "HcalDigiTask_sum_over_digis_fC_noise_%s", sub);
  meSumDigis_noise = book1D(histo, HistLim(50, -sumWithNoiseMax, sumWithNoiseMax));

  sprintf (histo, "HcalDigiTask_pedestal_%s", sub);
  mePedestal= book1D(histo, pedestalPlot);
  sprintf (histo, "HcalDigiTask_bin_5_frac_%s", sub);
  meBin5Frac = book1D(histo, fracPlot);
  sprintf (histo, "HcalDigiTask_bin_6_7_frac_%s", sub);
  meBin67Frac = book1D(histo, fracPlot);

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



