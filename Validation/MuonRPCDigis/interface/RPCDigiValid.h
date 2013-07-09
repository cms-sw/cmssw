#ifndef RPCDigiValid_h
#define RPCDigiValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

class RPCDigiValid: public edm::EDAnalyzer
{

public:

  RPCDigiValid(const edm::ParameterSet& ps);
  ~RPCDigiValid();

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginJob();
  void endJob(void);

private:

  MonitorElement* xyview;
  MonitorElement* rzview;
  MonitorElement* Res;
  MonitorElement* ResWmin2;
  MonitorElement* ResWmin1;
  MonitorElement* ResWzer0;
  MonitorElement* ResWplu1;
  MonitorElement* ResWplu2;
  MonitorElement* BxDist;
  MonitorElement* StripProf;

  MonitorElement* BxDist_whMin2;
  MonitorElement* BxDist_whMin1;
  MonitorElement* BxDist_wh0;
  MonitorElement* BxDist_wh0_st1;
  MonitorElement* BxDist_whPlu1;
  MonitorElement* BxDist_whPlu2;

  //barrel layers residuals
  MonitorElement* ResLayer1_barrel;
  MonitorElement* ResLayer2_barrel;
  MonitorElement* ResLayer3_barrel;
  MonitorElement* ResLayer4_barrel;
  MonitorElement* ResLayer5_barrel;
  MonitorElement* ResLayer6_barrel;

  //members for EndCap's disks:
  MonitorElement* ResDmin1;
  MonitorElement* ResDmin2;
  MonitorElement* ResDmin3;
  MonitorElement* ResDplu1;
  MonitorElement* ResDplu2;
  MonitorElement* ResDplu3;

  //endcap layters residuals
  MonitorElement* Res_Endcap1_Ring2_A;
  MonitorElement* Res_Endcap1_Ring2_B;
  MonitorElement* Res_Endcap1_Ring2_C;

  MonitorElement* Res_Endcap23_Ring2_A;
  MonitorElement* Res_Endcap23_Ring2_B;
  MonitorElement* Res_Endcap23_Ring2_C;

  MonitorElement* Res_Endcap123_Ring3_A;
  MonitorElement* Res_Endcap123_Ring3_B;
  MonitorElement* Res_Endcap123_Ring3_C;

  //new member for cls
  MonitorElement* noiseCLS;

  MonitorElement* clsBarrel;
  MonitorElement* clsLayer1;
  MonitorElement* clsLayer2;
  MonitorElement* clsLayer3;
  MonitorElement* clsLayer4;
  MonitorElement* clsLayer5;
  MonitorElement* clsLayer6;

  //CLS Validation
  //ring2, disc +- 1
  MonitorElement* CLS_Endcap_1_Ring2_A;
  MonitorElement* CLS_Endcap_1_Ring2_B;
  MonitorElement* CLS_Endcap_1_Ring2_C;

  //ring2, disc +-2 & +-3
  MonitorElement* CLS_Endcap_23_Ring2_A;
  MonitorElement* CLS_Endcap_23_Ring2_B;
  MonitorElement* CLS_Endcap_23_Ring2_C;

  //ring 3, all discs
  MonitorElement* CLS_Endcap_123_Ring3_A;
  MonitorElement* CLS_Endcap_123_Ring3_B;
  MonitorElement* CLS_Endcap_123_Ring3_C;
  //CLS Validation

  //new members for the noise
  std::map<RPCDetId, double> mapRollCls;
  std::map<RPCDetId, double> mapRollArea;
  std::map<RPCDetId, double> mapRollStripArea;
  std::map<RPCDetId, int> mapRollFakeCount;
  std::map<RPCDetId, int> mapRollTruCount;
  std::map<RPCDetId, std::string> mapRollName;
  std::map<RPCDetId, std::map<int, double>*> mapRollStripRate;
  std::map<RPCDetId, std::map<int, double>*> mapRollNoisyStripRate;
  int countEvent;

  DQMStore* dbe_;
  std::string outputFile_;
  std::string digiLabel;
};

#endif
