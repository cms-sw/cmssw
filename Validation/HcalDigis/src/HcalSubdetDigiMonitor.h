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
  void fillDigiSimhitProfile(double v1, double v2) { fillElements(meDigiSimhitProfile, v1, v2);}
  void fillNDigis(double value) {fillElement(menDigis, value);}
  void fillSumDigis(double value) {fillElement(meSumDigis, value);}
  
  void fillBin5Frac(double value) {fillElement(meBin5Frac, value);}
  void fillBin67Frac(double value) {fillElement(meBin67Frac, value);}
  void fillBin4567Frac(double value) {fillElement(meBin4567Frac, value);}  
  void fillPedestalfC(double value) {fillElement(mePedestalfC, value);}
  void fillADC0fC(double value) {fillElement(meDigifC, value);}
  void fillADC0count(double value) {fillElement(meADC0, value);}
  void fillDigiMinusPedfC(double value) {fillElement(meDigiMinusPedfC, value);}
  void fillPhiMC(double value) {fillElement(mePhiMC, value);}
  void fillEtaMC(double value) {fillElement(meEtaMC, value);}  
void fillNTowersGt10(double value) {fillElement(meNTowersGt10, value);}  
  void fillTimeSlice(double v1, double v2) {fillElements(meTimeSlice, v1,v2);}
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
  MonitorElement* meADC0;
  MonitorElement* meBin5Frac;
  MonitorElement* meBin67Frac;
  MonitorElement* meBin4567Frac;
  MonitorElement* mePedestalfC;
  MonitorElement* meDigifC;
  MonitorElement* meDigiMinusPedfC;
  MonitorElement* meEtaMC;
  MonitorElement* mePhiMC; 
  MonitorElement* meNTowersGt10;
  MonitorElement* meTimeSlice;
};

#endif

