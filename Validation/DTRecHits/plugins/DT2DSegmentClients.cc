#include "DT2DSegmentClients.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "utils.h"

//#include "TFile.h"
#include <string>
#include <iostream>
#include <map>


using namespace std;
using namespace edm;

DT2DSegmentClients::DT2DSegmentClients(const edm::ParameterSet& ps){
}

void DT2DSegmentClients::analyze(const Event& e, const EventSetup& context){
  Tutils * util = new Tutils;
  dbe = Service<DQMStore>().operator->();
  dbe->setCurrentFolder("DQMData/Run 1/DT/Run summary/2DSegments/");
  Tutils * aux = new Tutils;
  MonitorElement * hResPos = dbe->get("2D_SuperPhi_hResPos");
  MonitorElement * hResAngle = dbe->get("2D_SuperPhi_hResAngle");
  MonitorElement * hPullPos = dbe->get("2D_SuperPhi_hPullPos");
  MonitorElement * hPullAngle = dbe->get("2D_SuperPhi_hPullAngle");
  aux->drawGFit(hResPos->getTH1(),-0.1,0.1,-0.1,0.1);
  aux->drawGFit(hResAngle->getTH1(),-0.1,0.1,-0.1,0.1);
  aux->drawGFit(hPullPos->getTH1(),-5,5,-5,5);
  aux->drawGFit(hPullAngle->getTH1(),-5,5,-5,5);
  
}


