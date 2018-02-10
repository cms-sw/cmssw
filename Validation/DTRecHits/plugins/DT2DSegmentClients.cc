#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/DTRecHits/interface/utils.h"

#include "DT2DSegmentClients.h"

using namespace std;
using namespace edm;

DT2DSegmentClients::DT2DSegmentClients(edm::ParameterSet const& config)
{
}

void DT2DSegmentClients::endLuminosityBlock(edm::LuminosityBlock const& lumi,
  edm::EventSetup const& setup)
{
  DQMStore* dbe = Service<DQMStore>().operator->();
  MonitorElement * hResPos    = dbe->get("DT/2DSegments/Res/2D_SuperPhi_hResPos");
  MonitorElement * hResAngle  = dbe->get("DT/2DSegments/Res/2D_SuperPhi_hResAngle");
  MonitorElement * hPullPos   = dbe->get("DT/2DSegments/Pull/2D_SuperPhi_hPullPos");
  MonitorElement * hPullAngle = dbe->get("DT/2DSegments/Pull/2D_SuperPhi_hPullAngle");

  Tutils util;
  util.drawGFit(hResPos->getTH1(), -0.1, 0.1, -0.1, 0.1);
  util.drawGFit(hResAngle->getTH1(), -0.1, 0.1, -0.1, 0.1);
  util.drawGFit(hPullPos->getTH1(), -5, 5, -5, 5);
  util.drawGFit(hPullAngle->getTH1(), -5, 5, -5, 5);
}

void DT2DSegmentClients::analyze(Event const& event, EventSetup const& setup)
{
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DT2DSegmentClients);
