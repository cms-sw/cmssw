#ifndef HcalDigis_HcalSubdetDigiMonitor_h
#define HcalDigis_HcalSubdetDigiMonitor_h

#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"

class HcalSubdetDigiMonitor
{
public:

  HcalSubdetDigiMonitor(DQMStore* dbe, const std::string & subdet, int noise);

  // Ndigis 
  void fillmeNdigis(double v1)
  {fillElement(meNdigis, v1);}

  // occupancies filling
  void fillmeOccupancy_map_depth1(double v1, double v2)
  {fillElements(meOccupancy_map_depth1, v1, v2);}
  void fillmeOccupancy_map_depth2(double v1, double v2)
  {fillElements(meOccupancy_map_depth2, v1, v2);}
  void fillmeOccupancy_map_depth3(double v1, double v2)
  {fillElements(meOccupancy_map_depth3, v1, v2);}
  void fillmeOccupancy_map_depth4(double v1, double v2)
  {fillElements(meOccupancy_map_depth4, v1, v2);}

  void fillmeOccupancy_vs_ieta_depth1(double v1, double v2)
  {fillElements(meOccupancy_vs_ieta_depth1, v1, v2);}
  void fillmeOccupancy_vs_ieta_depth2(double v1, double v2)
  {fillElements(meOccupancy_vs_ieta_depth2, v1, v2);}
  void fillmeOccupancy_vs_ieta_depth3(double v1, double v2)
  {fillElements(meOccupancy_vs_ieta_depth3, v1, v2);}
  void fillmeOccupancy_vs_ieta_depth4(double v1, double v2)
  {fillElements(meOccupancy_vs_ieta_depth4, v1, v2);}

  // occupancies handling
  double getBinContent_depth1(int i, int j)
  {return getMeElementBinContent(meOccupancy_map_depth1, i, j);} 
  double getBinContent_depth2(int i, int j)
  {return getMeElementBinContent(meOccupancy_map_depth2, i, j);} 
  double getBinContent_depth3(int i, int j)
  {return getMeElementBinContent(meOccupancy_map_depth3, i, j);} 
  double getBinContent_depth4(int i, int j)
  {return getMeElementBinContent(meOccupancy_map_depth4, i, j);} 

  void setBinContent_depth1(int i, int j, double v)
  {setMeElementBinContent(meOccupancy_map_depth1, i, j, v);} 
  void setBinContent_depth2(int i, int j, double v)
  {setMeElementBinContent(meOccupancy_map_depth2, i, j, v);} 
  void setBinContent_depth3(int i, int j, double v)
  {setMeElementBinContent(meOccupancy_map_depth3, i, j, v);} 
  void setBinContent_depth4(int i, int j, double v)
  {setMeElementBinContent(meOccupancy_map_depth4, i, j, v);} 


  //
  void fillmeAmplIetaIphi1(double v1, double v2, double v3)
  {fillElements(meAmplIetaIphi1, v1, v2, v3);}
  void fillmeAmplIetaIphi2(double v1, double v2, double v3)
  {fillElements(meAmplIetaIphi2, v1, v2, v3);}
  void fillmeAmplIetaIphi3(double v1, double v2, double v3)
  {fillElements(meAmplIetaIphi3, v1, v2, v3);}
  void fillmeAmplIetaIphi4(double v1, double v2, double v3)
  {fillElements(meAmplIetaIphi4, v1, v2, v3);}  


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

  void fillmeAll10slices_depth1(double v1, double v2)
  {fillElements(meAll10slices_depth1, v1, v2);}
  void fillmeAll10slices_depth2(double v1, double v2)
  {fillElements(meAll10slices_depth2, v1, v2);}

  void fillmeAll10slices1D_depth1(double v1, double v2)
  {fillElements(meAll10slices1D_depth1, v1, v2);}
  void fillmeAll10slices1D_depth2(double v1, double v2)
  {fillElements(meAll10slices1D_depth2, v1, v2);}

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


  // DB constants 

  void fillmeGain0Depth1(double v1)
  {fillElement(meGain0Depth1, v1);}
  void fillmeGain1Depth1(double v1)
  {fillElement(meGain1Depth1, v1);}
  void fillmeGain2Depth1(double v1)
  {fillElement(meGain2Depth1, v1);}
  void fillmeGain3Depth1(double v1)
  {fillElement(meGain3Depth1, v1);}
  void fillmeGain0Depth2(double v1)
  {fillElement(meGain0Depth2, v1);}
  void fillmeGain1Depth2(double v1)
  {fillElement(meGain1Depth2, v1);}
  void fillmeGain2Depth2(double v1)
  {fillElement(meGain2Depth2, v1);}
  void fillmeGain3Depth2(double v1)
  {fillElement(meGain3Depth2, v1);}
  void fillmeGain0Depth3(double v1)
  {fillElement(meGain0Depth3, v1);}
  void fillmeGain1Depth3(double v1)
  {fillElement(meGain1Depth3, v1);}
  void fillmeGain2Depth3(double v1)
  {fillElement(meGain2Depth3, v1);}
  void fillmeGain3Depth3(double v1)
  {fillElement(meGain3Depth3, v1);}
  void fillmeGain0Depth4(double v1)
  {fillElement(meGain0Depth4, v1);}
  void fillmeGain1Depth4(double v1)
  {fillElement(meGain1Depth4, v1);}
  void fillmeGain2Depth4(double v1)
  {fillElement(meGain2Depth4, v1);}
  void fillmeGain3Depth4(double v1)
  {fillElement(meGain3Depth4, v1);}

  void fillmeGainWidth0Depth1(double v1)
  {fillElement(meGainWidth0Depth1, v1);}
  void fillmeGainWidth1Depth1(double v1)
  {fillElement(meGainWidth1Depth1, v1);}
  void fillmeGainWidth2Depth1(double v1)
  {fillElement(meGainWidth2Depth1, v1);}
  void fillmeGainWidth3Depth1(double v1)
  {fillElement(meGainWidth3Depth1, v1);}
  void fillmeGainWidth0Depth2(double v1)
  {fillElement(meGainWidth0Depth2, v1);}
  void fillmeGainWidth1Depth2(double v1)
  {fillElement(meGainWidth1Depth2, v1);}
  void fillmeGainWidth2Depth2(double v1)
  {fillElement(meGainWidth2Depth2, v1);}
  void fillmeGainWidth3Depth2(double v1)
  {fillElement(meGainWidth3Depth2, v1);}
  void fillmeGainWidth0Depth3(double v1)
  {fillElement(meGainWidth0Depth3, v1);}
  void fillmeGainWidth1Depth3(double v1)
  {fillElement(meGainWidth1Depth3, v1);}
  void fillmeGainWidth2Depth3(double v1)
  {fillElement(meGainWidth2Depth3, v1);}
  void fillmeGainWidth3Depth3(double v1)
  {fillElement(meGainWidth3Depth3, v1);}
  void fillmeGainWidth0Depth4(double v1)
  {fillElement(meGainWidth0Depth4, v1);}
  void fillmeGainWidth1Depth4(double v1)
  {fillElement(meGainWidth1Depth4, v1);}
  void fillmeGainWidth2Depth4(double v1)
  {fillElement(meGainWidth2Depth4, v1);}
  void fillmeGainWidth3Depth4(double v1)
  {fillElement(meGainWidth3Depth4, v1);}

  void fillmePed0Depth1(double v1)
  {fillElement(mePed0Depth1, v1);}
  void fillmePed1Depth1(double v1)
  {fillElement(mePed1Depth1, v1);}
  void fillmePed2Depth1(double v1)
  {fillElement(mePed2Depth1, v1);}
  void fillmePed3Depth1(double v1)
  {fillElement(mePed3Depth1, v1);}
  void fillmePed0Depth2(double v1)
  {fillElement(mePed0Depth2, v1);}
  void fillmePed1Depth2(double v1)
  {fillElement(mePed1Depth2, v1);}
  void fillmePed2Depth2(double v1)
  {fillElement(mePed2Depth2, v1);}
  void fillmePed3Depth2(double v1)
  {fillElement(mePed3Depth2, v1);}
  void fillmePed0Depth3(double v1)
  {fillElement(mePed0Depth3, v1);}
  void fillmePed1Depth3(double v1)
  {fillElement(mePed1Depth3, v1);}
  void fillmePed2Depth3(double v1)
  {fillElement(mePed2Depth3, v1);}
  void fillmePed3Depth3(double v1)
  {fillElement(mePed3Depth3, v1);}
  void fillmePed0Depth4(double v1)
  {fillElement(mePed0Depth4, v1);}
  void fillmePed1Depth4(double v1)
  {fillElement(mePed1Depth4, v1);}
  void fillmePed2Depth4(double v1)
  {fillElement(mePed2Depth4, v1);}
  void fillmePed3Depth4(double v1)
  {fillElement(mePed3Depth4, v1);}


  void fillmePedWidth0Depth1(double v1)
  {fillElement(mePedWidth0Depth1, v1);}
  void fillmePedWidth1Depth1(double v1)
  {fillElement(mePedWidth1Depth1, v1);}
  void fillmePedWidth2Depth1(double v1)
  {fillElement(mePedWidth2Depth1, v1);}
  void fillmePedWidth3Depth1(double v1)
  {fillElement(mePedWidth3Depth1, v1);}
  void fillmePedWidth0Depth2(double v1)
  {fillElement(mePedWidth0Depth2, v1);}
  void fillmePedWidth1Depth2(double v1)
  {fillElement(mePedWidth1Depth2, v1);}
  void fillmePedWidth2Depth2(double v1)
  {fillElement(mePedWidth2Depth2, v1);}
  void fillmePedWidth3Depth2(double v1)
  {fillElement(mePedWidth3Depth2, v1);}
  void fillmePedWidth0Depth3(double v1)
  {fillElement(mePedWidth0Depth3, v1);}
  void fillmePedWidth1Depth3(double v1)
  {fillElement(mePedWidth1Depth3, v1);}
  void fillmePedWidth2Depth3(double v1)
  {fillElement(mePedWidth2Depth3, v1);}
  void fillmePedWidth3Depth3(double v1)
  {fillElement(mePedWidth3Depth3, v1);}
  void fillmePedWidth0Depth4(double v1)
  {fillElement(mePedWidth0Depth4, v1);}
  void fillmePedWidth1Depth4(double v1)
  {fillElement(mePedWidth1Depth4, v1);}
  void fillmePedWidth2Depth4(double v1)
  {fillElement(mePedWidth2Depth4, v1);}
  void fillmePedWidth3Depth4(double v1)
  {fillElement(mePedWidth3Depth4, v1);}

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

  double getMeElementBinContent(MonitorElement* me, int i, int j)
  {
    double tmp = 0.;
    if(me) tmp = me->getBinContent(i,j);
    return tmp;
  }

  void setMeElementBinContent(MonitorElement* me, int i, int j, double v)
  {
    if(me) me->setBinContent(i,j,v);
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

  MonitorElement* meNdigis;

  MonitorElement* meOccupancy_map_depth1;
  MonitorElement* meOccupancy_map_depth2;
  MonitorElement* meOccupancy_map_depth3;
  MonitorElement* meOccupancy_map_depth4;
  MonitorElement* meOccupancy_vs_ieta_depth1;
  MonitorElement* meOccupancy_vs_ieta_depth2;
  MonitorElement* meOccupancy_vs_ieta_depth3;
  MonitorElement* meOccupancy_vs_ieta_depth4;


  MonitorElement* meAmplIetaIphi1;
  MonitorElement* meAmplIetaIphi2;
  MonitorElement* meAmplIetaIphi3;
  MonitorElement* meAmplIetaIphi4;
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
  MonitorElement* meAll10slices_depth1;
  MonitorElement* meAll10slices_depth2;
  MonitorElement* meAll10slices1D_depth1;
  MonitorElement* meAll10slices1D_depth2;
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

  // DB constants 

  MonitorElement* meGain0Depth1;
  MonitorElement* meGain1Depth1;
  MonitorElement* meGain2Depth1;
  MonitorElement* meGain3Depth1;
  MonitorElement* meGain0Depth2;
  MonitorElement* meGain1Depth2;
  MonitorElement* meGain2Depth2;
  MonitorElement* meGain3Depth2;
  MonitorElement* meGain0Depth3;
  MonitorElement* meGain1Depth3;
  MonitorElement* meGain2Depth3;
  MonitorElement* meGain3Depth3;
  MonitorElement* meGain0Depth4;
  MonitorElement* meGain1Depth4;
  MonitorElement* meGain2Depth4;
  MonitorElement* meGain3Depth4;

  MonitorElement* meGainWidth0Depth1;
  MonitorElement* meGainWidth1Depth1;
  MonitorElement* meGainWidth2Depth1;
  MonitorElement* meGainWidth3Depth1;
  MonitorElement* meGainWidth0Depth2;
  MonitorElement* meGainWidth1Depth2;
  MonitorElement* meGainWidth2Depth2;
  MonitorElement* meGainWidth3Depth2;
  MonitorElement* meGainWidth0Depth3;
  MonitorElement* meGainWidth1Depth3;
  MonitorElement* meGainWidth2Depth3;
  MonitorElement* meGainWidth3Depth3;
  MonitorElement* meGainWidth0Depth4;
  MonitorElement* meGainWidth1Depth4;
  MonitorElement* meGainWidth2Depth4;
  MonitorElement* meGainWidth3Depth4;

  MonitorElement* mePed0Depth1;
  MonitorElement* mePed1Depth1;
  MonitorElement* mePed2Depth1;
  MonitorElement* mePed3Depth1;
  MonitorElement* mePed0Depth2;
  MonitorElement* mePed1Depth2;
  MonitorElement* mePed2Depth2;
  MonitorElement* mePed3Depth2;
  MonitorElement* mePed0Depth3;
  MonitorElement* mePed1Depth3;
  MonitorElement* mePed2Depth3;
  MonitorElement* mePed3Depth3;
  MonitorElement* mePed0Depth4;
  MonitorElement* mePed1Depth4;
  MonitorElement* mePed2Depth4;
  MonitorElement* mePed3Depth4;

  MonitorElement* mePedWidth0Depth1;
  MonitorElement* mePedWidth1Depth1;
  MonitorElement* mePedWidth2Depth1;
  MonitorElement* mePedWidth3Depth1;
  MonitorElement* mePedWidth0Depth2;
  MonitorElement* mePedWidth1Depth2;
  MonitorElement* mePedWidth2Depth2;
  MonitorElement* mePedWidth3Depth2;
  MonitorElement* mePedWidth0Depth3;
  MonitorElement* mePedWidth1Depth3;
  MonitorElement* mePedWidth2Depth3;
  MonitorElement* mePedWidth3Depth3;
  MonitorElement* mePedWidth0Depth4;
  MonitorElement* mePedWidth1Depth4;
  MonitorElement* mePedWidth2Depth4;
  MonitorElement* mePedWidth3Depth4;


  // CapID-0 only 
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

