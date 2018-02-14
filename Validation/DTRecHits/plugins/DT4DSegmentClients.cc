#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/DTRecHits/interface/utils.h"

#include "DT4DSegmentClients.h"

using namespace std;
using namespace edm;

DT4DSegmentClients::DT4DSegmentClients(edm::ParameterSet const& config)
{
}

void DT4DSegmentClients::endLuminosityBlock(edm::LuminosityBlock const& lumi,
  edm::EventSetup const& setup)
{
  DQMStore* dbe = Service<DQMStore>().operator->();
  
  MonitorElement * hResAlpha = dbe->get("DT/4DSegments/Res/4D_All_hResAlpha");
  MonitorElement * hResBeta = dbe->get("DT/4DSegments/Res/4D_All_hResBeta");
  MonitorElement * hResX = dbe->get("DT/4DSegments/Res/4D_All_hResX");
  MonitorElement * hResY = dbe->get("DT/4DSegments/Res/4D_All_hResY");
  MonitorElement * hResBetaRZ = dbe->get("DT/4DSegments/Res/4D_All_hResBetaRZ");
  MonitorElement * hResYRZ = dbe->get("DT/4DSegments/Res/4D_All_hResYRZ");

  MonitorElement * hResAlpha_W0 = dbe->get("DT/4DSegments/Res/4D_W0_hResAlpha");
  MonitorElement * hResBeta_W0 = dbe->get("DT/4DSegments/Res/4D_W0_hResBeta");
  MonitorElement * hResX_W0 = dbe->get("DT/4DSegments/Res/4D_W0_hResX");
  MonitorElement * hResY_W0 = dbe->get("DT/4DSegments/Res/4D_W0_hResY");
  MonitorElement * hResBetaRZ_W0 = dbe->get("DT/4DSegments/Res/4D_W0_hResBetaRZ");
  MonitorElement * hResYRZ_W0 = dbe->get("DT/4DSegments/Res/4D_W0_hResYRZ");

  MonitorElement * hResAlpha_W1 = dbe->get("DT/4DSegments/Res/4D_W1_hResAlpha");
  MonitorElement * hResBeta_W1 = dbe->get("DT/4DSegments/Res/4D_W1_hResBeta");
  MonitorElement * hResX_W1 = dbe->get("DT/4DSegments/Res/4D_W1_hResX");
  MonitorElement * hResY_W1 = dbe->get("DT/4DSegments/Res/4D_W1_hResY");
  MonitorElement * hResBetaRZ_W1 = dbe->get("DT/4DSegments/Res/4D_W1_hResBetaRZ");
  MonitorElement * hResYRZ_W1 = dbe->get("DT/4DSegments/Res/4D_W1_hResYRZ");

  MonitorElement * hResAlpha_W2 = dbe->get("DT/4DSegments/Res/4D_W2_hResAlpha");
  MonitorElement * hResBeta_W2 = dbe->get("DT/4DSegments/Res/4D_W2_hResBeta");
  MonitorElement * hResX_W2 = dbe->get("DT/4DSegments/Res/4D_W2_hResX");
  MonitorElement * hResY_W2 = dbe->get("DT/4DSegments/Res/4D_W2_hResY");
  MonitorElement * hResBetaRZ_W2 = dbe->get("DT/4DSegments/Res/4D_W2_hResBetaRZ");
  MonitorElement * hResYRZ_W2 = dbe->get("DT/4DSegments/Res/4D_W2_hResYRZ");

  MonitorElement * hPullAlpha = dbe->get("DT/4DSegments/Pull/4D_All_hPullAlpha");
  MonitorElement * hPullBeta = dbe->get("DT/4DSegments/Pull/4D_All_hPullBeta");
  MonitorElement * hPullX = dbe->get("DT/4DSegments/Pull/4D_All_hPullX");
  MonitorElement * hPullY = dbe->get("DT/4DSegments/Pull/4D_All_hPullY");
  MonitorElement * hPullBetaRZ = dbe->get("DT/4DSegments/Pull/4D_All_hPullBetaRZ");
  MonitorElement * hPullYRZ = dbe->get("DT/4DSegments/Pull/4D_All_hPullYRZ");

  MonitorElement * hPullAlpha_W0 = dbe->get("DT/4DSegments/Pull/4D_W0_hPullAlpha");
  MonitorElement * hPullBeta_W0 = dbe->get("DT/4DSegments/Pull/4D_W0_hPullBeta");
  MonitorElement * hPullX_W0 = dbe->get("DT/4DSegments/Pull/4D_W0_hPullX");
  MonitorElement * hPullY_W0 = dbe->get("DT/4DSegments/Pull/4D_W0_hPullY");
  MonitorElement * hPullBetaRZ_W0 = dbe->get("DT/4DSegments/Pull/4D_W0_hPullBetaRZ");
  MonitorElement * hPullYRZ_W0 = dbe->get("DT/4DSegments/Pull/4D_W0_hPullYRZ");

  MonitorElement * hPullAlpha_W1 = dbe->get("DT/4DSegments/Pull/4D_W1_hPullAlpha");
  MonitorElement * hPullBeta_W1 = dbe->get("DT/4DSegments/Pull/4D_W1_hPullBeta");
  MonitorElement * hPullX_W1 = dbe->get("DT/4DSegments/Pull/4D_W1_hPullX");
  MonitorElement * hPullY_W1 = dbe->get("DT/4DSegments/Pull/4D_W1_hPullY");
  MonitorElement * hPullBetaRZ_W1 = dbe->get("DT/4DSegments/Pull/4D_W1_hPullBetaRZ");
  MonitorElement * hPullYRZ_W1 = dbe->get("DT/4DSegments/Pull/4D_W1_hPullYRZ");

  MonitorElement * hPullAlpha_W2 = dbe->get("DT/4DSegments/Pull/4D_W2_hPullAlpha");
  MonitorElement * hPullBeta_W2 = dbe->get("DT/4DSegments/Pull/4D_W2_hPullBeta");
  MonitorElement * hPullX_W2 = dbe->get("DT/4DSegments/Pull/4D_W2_hPullX");
  MonitorElement * hPullY_W2 = dbe->get("DT/4DSegments/Pull/4D_W2_hPullY");
  MonitorElement * hPullBetaRZ_W2 = dbe->get("DT/4DSegments/Pull/4D_W2_hPullBetaRZ");
  MonitorElement * hPullYRZ_W2 = dbe->get("DT/4DSegments/Pull/4D_W2_hPullYRZ");
  
  Tutils util;
  util.drawGFit(hResAlpha->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBeta->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResX->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResY->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBetaRZ->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResYRZ->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hResAlpha_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBeta_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResX_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResY_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBetaRZ_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResYRZ_W0->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hResAlpha_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBeta_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResX_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResY_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBetaRZ_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResYRZ_W1->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hResAlpha_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBeta_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResX_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResY_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResBetaRZ_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hResYRZ_W2->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hPullAlpha->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBeta->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullX->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullY->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBetaRZ->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullYRZ->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hPullAlpha_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBeta_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullX_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullY_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBetaRZ_W0->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullYRZ_W0->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hPullAlpha_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBeta_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullX_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullY_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBetaRZ_W1->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullYRZ_W1->getTH1(),-0.2,0.2,-0.1,0.1);

  util.drawGFit(hPullAlpha_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBeta_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullX_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullY_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullBetaRZ_W2->getTH1(),-0.2,0.2,-0.1,0.1);
  util.drawGFit(hPullYRZ_W2->getTH1(),-0.2,0.2,-0.1,0.1);
}

void DT4DSegmentClients::analyze(Event const& event, EventSetup const& setup)
{
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DT4DSegmentClients);
