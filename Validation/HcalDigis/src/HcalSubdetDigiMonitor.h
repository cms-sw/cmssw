#ifndef HcalDigis_HcalSubdetDigiMonitor_h
#define HcalDigis_HcalSubdetDigiMonitor_h

#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"

class HcalSubdetDigiMonitor
{
public:

  HcalSubdetDigiMonitor(DQMStore* dbe, const std::string & subdet, int noise);
  void fillmeAmplIetaIphi(double v1, double v2, double v3)
  {fillElements(meAmplIetaIphi, v1, v2, v3);}
  
  void fillmeSumAmp(double v1)
  {fillElement(meSumAmp, v1);}

  void fillmenDigis(double v1)
  {fillElement(menDigis, v1 );}

  void fillmeADC0_depth1(double v1)
  {fillElement(meADC0_depth1, v1 );}
  void fillmeADC0_depth2(double v1)
  {fillElement(meADC0_depth2, v1 );}
  void fillmeADC0_depth3(double v1)
  {fillElement(meADC0_depth3, v1);}
  void fillmeADC0_depth4(double v1)
  {fillElement(meADC0_depth4, v1);}

  void fillmeADC0fC_depth1(double v1)
  {fillElement(meADC0fC_depth1, v1);}
  void fillmeADC0fC_depth2(double v1)
  {fillElement(meADC0fC_depth2, v1);}
  void fillmeADC0fC_depth3(double v1)
  {fillElement(meADC0fC_depth3, v1);}
  void fillmeADC0fC_depth4(double v1)
  {fillElement(meADC0fC_depth4, v1);}

  void fillmeSignalAmp(double v1)
  {fillElement(meSignalAmp, v1);}
  void fillmeSignalAmp1(double v1)
  {fillElement(meSignalAmp1, v1);}
  void fillmeSignalAmp2(double v1)
  {fillElement(meSignalAmp2, v1);}
  void fillmeSignalAmp3(double v1)
  {fillElement(meSignalAmp3, v1);}
  void fillmeSignalAmp4(double v1)
  {fillElement(meSignalAmp4, v1);}

  void fillmeSignalTimeSlice(double v1, double v2)
  {fillElements(meSignalTimeSlice, v1, v2);}

  void fillmeAll10slices(double v1, double v2)
  {fillElements(meAll10slices, v1, v2);}

  void fillmeBin5Frac(double v1)
  {fillElement(meBin5Frac, v1);}
  void fillmeBin67Frac(double v1)
  {fillElement(meBin67Frac, v1);}

  void fillmeDigiSimhit(double v1, double v2)
  {fillElements(meDigiSimhit, v1, v2);}
  void fillmeDigiSimhit1(double v1, double v2)
  {fillElements(meDigiSimhit1, v1, v2);}
  void fillmeDigiSimhit2(double v1, double v2)
  {fillElements(meDigiSimhit2, v1, v2);}
  void fillmeDigiSimhit3(double v1, double v2)
  {fillElements(meDigiSimhit3, v1, v2);}
  void fillmeDigiSimhit4(double v1, double v2)
  {fillElements(meDigiSimhit4, v1, v2);}

  void fillmeDigiSimhitProfile(double v1, double v2)
  {fillElements(meDigiSimhitProfile, v1, v2);}
  void fillmeDigiSimhitProfile1(double v1, double v2)
  {fillElements(meDigiSimhitProfile1, v1, v2);}
  void fillmeDigiSimhitProfile2(double v1, double v2)
  {fillElements(meDigiSimhitProfile2, v1, v2);}
  void fillmeDigiSimhitProfile3(double v1, double v2)
  {fillElements(meDigiSimhitProfile3, v1, v2);}
  void fillmeDigiSimhitProfile4(double v1, double v2)
  {fillElements(meDigiSimhitProfile4, v1, v2);}

  void fillmeRatioDigiSimhit(double v1)
  {fillElement(meRatioDigiSimhit, v1);}
  void fillmeRatioDigiSimhit1(double v1)
  {fillElement(meRatioDigiSimhit1, v1);}
  void fillmeRatioDigiSimhit2(double v1)
  {fillElement(meRatioDigiSimhit2, v1);}
  void fillmeRatioDigiSimhit3(double v1)
  {fillElement(meRatioDigiSimhit3, v1);}
  void fillmeRatioDigiSimhit4(double v1)
  {fillElement(meRatioDigiSimhit4, v1);}

  void fillmePedCapId0(double v1)
  {fillElement(mePedCapId0, v1);}
  void fillmePedCapId1(double v1)
  {fillElement(mePedCapId1, v1);}
  void fillmePedCapId2(double v1)
  {fillElement(mePedCapId2, v1);}
  void fillmePedCapId3(double v1)
  {fillElement(mePedCapId3, v1);}

  void fillmePedWidthCapId0(double v1)
  {fillElement(mePedWidthCapId0, v1);}
  void fillmePedWidthCapId1(double v1)
  {fillElement(mePedWidthCapId1, v1);}
  void fillmePedWidthCapId2(double v1)
  {fillElement(mePedWidthCapId2, v1);}
  void fillmePedWidthCapId3(double v1)
  {fillElement(mePedWidthCapId3, v1);}

  void fillmeGainDepth1(double v1)
  {fillElement(meGainDepth1, v1);}
  void fillmeGainDepth2(double v1)
  {fillElement(meGainDepth2, v1);}
  void fillmeGainDepth3(double v1)
  {fillElement(meGainDepth3, v1);}
  void fillmeGainDepth4(double v1)
  {fillElement(meGainDepth4, v1);}

  void fillmeGainWidthDepth1(double v1)
  {fillElement(meGainWidthDepth1, v1);}
  void fillmeGainWidthDepth2(double v1)
  {fillElement(meGainWidthDepth2, v1);}
  void fillmeGainWidthDepth3(double v1)
  {fillElement(meGainWidthDepth3, v1);}
  void fillmeGainWidthDepth4(double v1)
  {fillElement(meGainWidthDepth4, v1);}

  void fillmeGainMap1(double v1, double v2, double v3)
  {fillElements(meGainMap1, v1, v2, v3);}
  void fillmeGainMap2(double v1, double v2, double v3)
  {fillElements(meGainMap2, v1, v2, v3);}
  void fillmeGainMap3(double v1, double v2, double v3)
  {fillElements(meGainMap3, v1, v2, v3);}
  void fillmeGainMap4(double v1, double v2, double v3)
  {fillElements(meGainMap4, v1, v2, v3);}

  void fillmePwidthMap1(double v1, double v2, double v3)
  {fillElements(mePwidthMap1, v1, v2, v3);}
  void fillmePwidthMap2(double v1, double v2, double v3)
  {fillElements(mePwidthMap2, v1, v2, v3);}
  void fillmePwidthMap3(double v1, double v2, double v3)
  {fillElements(mePwidthMap3, v1, v2, v3);}
  void fillmePwidthMap4(double v1, double v2, double v3)
  {fillElements(mePwidthMap4, v1, v2, v3);}

private:

  void fillElement(MonitorElement* me, double value)
  {
    if(me) me->Fill(value);
  }
  void fillElements(MonitorElement* me, double v1, double v2)
  {
    if(me) me->Fill(v1, v2);
  }
  void fillElements(MonitorElement* me, double v1, double v2, double v3)
  {
    if(me) me->Fill(v1, v2, v3);
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

  DQMStore* dbe_;
  std::string subdet_;
  int noise_;

  MonitorElement* meAmplIetaIphi;
  MonitorElement* meSumAmp;
  MonitorElement* menDigis;

  MonitorElement* meADC0_depth1;
  MonitorElement* meADC0_depth2;
  MonitorElement* meADC0_depth3;
  MonitorElement* meADC0_depth4; 
  MonitorElement* meADC0fC_depth1;
  MonitorElement* meADC0fC_depth2;
  MonitorElement* meADC0fC_depth3;
  MonitorElement* meADC0fC_depth4;

  MonitorElement* meSignalAmp;
  MonitorElement* meSignalAmp1;
  MonitorElement* meSignalAmp2;
  MonitorElement* meSignalAmp3;
  MonitorElement* meSignalAmp4;

  MonitorElement* meSignalTimeSlice;
  MonitorElement* meAll10slices;
  MonitorElement* meBin5Frac;
  MonitorElement* meBin67Frac;

  MonitorElement* meDigiSimhit;
  MonitorElement* meDigiSimhit1;
  MonitorElement* meDigiSimhit2;
  MonitorElement* meDigiSimhit3;
  MonitorElement* meDigiSimhit4;

  MonitorElement* meRatioDigiSimhit;
  MonitorElement* meRatioDigiSimhit1;
  MonitorElement* meRatioDigiSimhit2;
  MonitorElement* meRatioDigiSimhit3;
  MonitorElement* meRatioDigiSimhit4;

  MonitorElement* meDigiSimhitProfile;
  MonitorElement* meDigiSimhitProfile1;
  MonitorElement* meDigiSimhitProfile2;
  MonitorElement* meDigiSimhitProfile3;
  MonitorElement* meDigiSimhitProfile4;

  MonitorElement* mePedCapId0;
  MonitorElement* mePedCapId1;
  MonitorElement* mePedCapId2;
  MonitorElement* mePedCapId3;
  MonitorElement* mePedWidthCapId0;
  MonitorElement* mePedWidthCapId1;
  MonitorElement* mePedWidthCapId2;
  MonitorElement* mePedWidthCapId3;

  MonitorElement* meGainDepth1;
  MonitorElement* meGainDepth2;
  MonitorElement* meGainDepth3;
  MonitorElement* meGainDepth4;
  MonitorElement* meGainWidthDepth1;
  MonitorElement* meGainWidthDepth2;
  MonitorElement* meGainWidthDepth3;
  MonitorElement* meGainWidthDepth4;

  MonitorElement* meGainMap1;
  MonitorElement* meGainMap2;
  MonitorElement* meGainMap3;
  MonitorElement* meGainMap4;

  MonitorElement* mePwidthMap1;
  MonitorElement* mePwidthMap2; 
  MonitorElement* mePwidthMap3; 
  MonitorElement* mePwidthMap4; 

};

#endif

