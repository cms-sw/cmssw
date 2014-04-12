#include "DT2DSegmentClients.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/DTRecHits/interface/utils.h"

//#include "TFile.h"
#include <string>
#include <iostream>
#include <map>


using namespace std;
using namespace edm;

DT2DSegmentClients::DT2DSegmentClients(const edm::ParameterSet& ps){
}
DT2DSegmentClients::~DT2DSegmentClients(){
}
void DT2DSegmentClients::endJob(){}

void DT2DSegmentClients::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
  edm::EventSetup const& c){
  dbe = Service<DQMStore>().operator->();
  //dbe->setCurrentFolder("/DT/2DSegments/");
  Tutils * aux = new Tutils;
  MonitorElement * hResPos = dbe->get("DT/2DSegments/Res/2D_SuperPhi_hResPos");
  MonitorElement * hResAngle = dbe->get("DT/2DSegments/Res/2D_SuperPhi_hResAngle");
  MonitorElement * hPullPos = dbe->get("DT/2DSegments/Pull/2D_SuperPhi_hPullPos");
  MonitorElement * hPullAngle = dbe->get("DT/2DSegments/Pull/2D_SuperPhi_hPullAngle");
  aux->drawGFit(hResPos->getTH1(),-0.1,0.1,-0.1,0.1);
  aux->drawGFit(hResAngle->getTH1(),-0.1,0.1,-0.1,0.1);
  aux->drawGFit(hPullPos->getTH1(),-5,5,-5,5);
  aux->drawGFit(hPullAngle->getTH1(),-5,5,-5,5);
  
}

void DT2DSegmentClients::analyze(const Event& e, const EventSetup& context){
}


