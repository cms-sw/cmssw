#ifndef HcalDigis_HcalSubdetDigiMonitor_h
#define HcalDigis_HcalSubdetDigiMonitor_h

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include <string>

class HcalSubdetDigiMonitor
{
public:

  HcalSubdetDigiMonitor(DaqMonitorBEInterface* dbe, const std::string & subdet);

  void fillEta(double value) {fillElement(meEta, value);}
  void fillPhi(double value) {fillElement(mePhi, value);}
  void fillDigiSimhit(double v1, double v2) {fillElements(meDigiSimhit, v1,v2);}
  void fillRatioDigiSimhit(double value) {fillElement(meRatioDigiSimhit, value);}
  void fillDigiSimhitProfile(double v1, double v2) {
    fillElements(meDigiSimhitProfile, v1, v2);}
  void fillNDigis(double value) {fillElement(menDigis, value);}
  void fillSumDigis(double value) {fillElement(meSumDigis, value);}
  void fillSumDigis_noise(double value) {fillElement(meSumDigis_noise, value);}
  void fillPedestal(double value) {fillElement(mePedestal, value);}
  void fillBin4Frac(double value) {fillElement(meBin4Frac, value);}
  void fillBin56Frac(double value) {fillElement(meBin56Frac, value);}

private:
  void fillElement(MonitorElement* me, double value)
  {
    if(me) me->Fill(value);
  }
  void fillElements(MonitorElement* me, double v1, double v2)
  {
    if(me) me->Fill(v1, v2);
  }

  // little embedded struct
  struct HistLim
  {
    HistLim(int nbin, float mini, float maxi)
    : n(nbin), min(mini), max(maxi) {}
    int n;
    float min;
    float max;
  };

  // utilities to create MonitorElements
  MonitorElement * book1D(const std::string & name, HistLim lim);
  MonitorElement * book2D(const std::string & name, HistLim lim1,
                                                    HistLim lim2);
  MonitorElement * bookProfile(const std::string & name, HistLim lim1, 
                                                         HistLim lim2);

  DaqMonitorBEInterface* dbe_;
  std::string subdet_;
  MonitorElement* meEta;
  MonitorElement* mePhi;
  MonitorElement* meDigiSimhit;
  MonitorElement* meRatioDigiSimhit;
  MonitorElement* meDigiSimhitProfile;
  MonitorElement* menDigis;
  MonitorElement* meSumDigis;
  MonitorElement* meSumDigis_noise;
  MonitorElement* mePedestal;
  MonitorElement* meBin4Frac;
  MonitorElement* meBin56Frac;
};

#endif

