#include "Validation/DTRecHits/plugins/DTRecHitClients.h"
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

DTRecHitClients::DTRecHitClients(const edm::ParameterSet& ps){
  dbe = Service<DQMStore>().operator->();
}
DTRecHitClients::~DTRecHitClients(){
}
void DTRecHitClients::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
    edm::EventSetup const& c){
  Tutils * util = new Tutils;
  /*hRes = dbe->get("DT/1DRecHits/1D_S1RPhi_hDist");
  cout << "Valores" << endl;
  cout << hRes->hasError() << endl;
  cout <<" " << hRes->getRMS() << endl;
  cout << "get th1" << endl;
  TH1 * h = hRes->getTH1();
  cout << h->GetEntries()<<endl;
  cout<<"***********************************"<<endl;*/
  MonitorElement * hRes_S3RPhi = dbe->get("DT/1DRecHits/Res/1D_S3RPhi_hRes");
  MonitorElement * hRes_S3RZ = dbe->get("DT/1DRecHits/Res/1D_S3RZ_hRes");
  MonitorElement * hRes_S3RZ_W0 = dbe->get("DT/1DRecHits/Res/1D_S3RZ_W0_hRes");
  MonitorElement * hRes_S3RZ_W1 = dbe->get("DT/1DRecHits/Res/1D_S3RZ_W1_hRes");
  MonitorElement * hRes_S3RZ_W2 = dbe->get("DT/1DRecHits/Res/1D_S3RZ_W2_hRes");

  MonitorElement * hPull_S3RPhi = dbe->get("DT/1DRecHits/Pull/1D_S3RPhi_hPull");
  MonitorElement * hPull_S3RZ = dbe->get("DT/1DRecHits/Pull/1D_S3RZ_hPull");
  MonitorElement * hPull_S3RZ_W0 = dbe->get("DT/1DRecHits/Pull/1D_S3RZ_W0_hPull");
  MonitorElement * hPull_S3RZ_W1 = dbe->get("DT/1DRecHits/Pull/1D_S3RZ_W1_hPull");
  MonitorElement * hPull_S3RZ_W2 = dbe->get("DT/1DRecHits/Pull/1D_S3RZ_W2_hPull");

  /*MonitorElement * hResVsEta_S3RPhi = dbe->get("1D_S3RPhi_hResVsEta");
  MonitorElement * hResVsEta_S3RZ = dbe->get("1D_S3RZ_hResVsEta");
  MonitorElement * hResVsEta_S3RZ_W0 = dbe->get("1D_S3RZ_W0_hResVsEta");
  MonitorElement * hResVsEta_S3RZ_W1 = dbe->get("1D_S3RZ_W1_hResVsEta");
  MonitorElement * hResVsEta_S3RZ_W2 = dbe->get("1D_S3RZ_W2_hResVsEta");

  MonitorElement * hResVsPhi_S3RPhi = dbe->get("1D_S3RPhi_hResVsPhi");
  MonitorElement * hResVsPhi_S3RZ = dbe->get("1D_S3RZ_hResVsPhi");
  MonitorElement * hResVsPhi_S3RZ_W0 = dbe->get("1D_S3RZ_W0_hResVsPhi");
  MonitorElement * hResVsPhi_S3RZ_W1 = dbe->get("1D_S3RZ_W1_hResVsPhi");
  MonitorElement * hResVsPhi_S3RZ_W2 = dbe->get("1D_S3RZ_W2_hResVsPhi");

  MonitorElement * hResVsPos_S3RPhi = dbe->get("1D_S3RPhi_hResVsPos");
  MonitorElement * hResVsPos_S3RZ = dbe->get("1D_S3RZ_hResVsPos");
  MonitorElement * hResVsPos_S3RZ_W0 = dbe->get("1D_S3RZ_W0_hResVsPos");
  MonitorElement * hResVsPos_S3RZ_W1 = dbe->get("1D_S3RZ_W1_hResVsPos");
  MonitorElement * hResVsPos_S3RZ_W2 = dbe->get("1D_S3RZ_W2_hResVsPos");*/

  util->drawGFit(hRes_S3RPhi->getTH1(),-0.2,0.2,-0.1,0.1);
  util->drawGFit(hRes_S3RZ->getTH1(),-0.2,0.2,-0.1,0.1);
  util->drawGFit(hRes_S3RZ_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util->drawGFit(hRes_S3RZ_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util->drawGFit(hRes_S3RZ_W2->getTH1(),-0.2,0.2,-0.1,0.1);

  util->drawGFit(hPull_S3RPhi->getTH1(),-5,5,-5,5);
  util->drawGFit(hPull_S3RZ->getTH1(),-5,5,-5,5);
  util->drawGFit(hPull_S3RZ_W0->getTH1(),-5,5,-5,5);
  util->drawGFit(hPull_S3RZ_W1->getTH1(),-5,5,-5,5);
  util->drawGFit(hPull_S3RZ_W2->getTH1(),-5,5,-5,5);

  /*util->plotAndProfileX(hResVsEta_S3RPhi->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsEta_S3RZ->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsEta_S3RZ_W0->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsEta_S3RZ_W1->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsEta_S3RZ_W2->getTH2F(),-0.6,0.6);

  util->plotAndProfileX(hResVsPhi_S3RPhi->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsPhi_S3RZ->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsPhi_S3RZ_W0->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsPhi_S3RZ_W1->getTH2F(),-0.6,0.6);
  util->plotAndProfileX(hResVsPhi_S3RZ_W2->getTH2F(),-0.6,0.6);

  util->plotAndProfileX(hResVsPos_S3RPhi->getTH2F(),-0.6,0.6,true);
  util->plotAndProfileX(hResVsPos_S3RZ->getTH2F(),-0.6,0.6,true);
  util->plotAndProfileX(hResVsPos_S3RZ_W0->getTH2F(),-0.6,0.6,true);
  util->plotAndProfileX(hResVsPos_S3RZ_W1->getTH2F(),-0.6,0.6,true);
  util->plotAndProfileX(hResVsPos_S3RZ_W2->getTH2F(),-0.6,0.6,true);
  cout<<"END***********************************"<<endl;

  */
}
void DTRecHitClients::endJob() {
}

void DTRecHitClients::analyze(const Event& e, const EventSetup& context){

  
}
