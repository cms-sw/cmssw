#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/DTRecHits/interface/utils.h"

#include "DTRecHitClients.h"
#include "Histograms.h"

using namespace std;
using namespace edm;

DTRecHitClients::DTRecHitClients(edm::ParameterSet const &pset) {
  // Switches for analysis at various steps
  doStep1_ = pset.getUntrackedParameter<bool>("doStep1", false);
  doStep2_ = pset.getUntrackedParameter<bool>("doStep2", false);
  doStep3_ = pset.getUntrackedParameter<bool>("doStep3", false);
  doall_ = pset.getUntrackedParameter<bool>("doall", false);
  local_ = pset.getUntrackedParameter<bool>("local", true);
}

DTRecHitClients::~DTRecHitClients() {}

void DTRecHitClients::dqmEndJob(DQMStore::IBooker &booker, DQMStore::IGetter &getter) {
  MonitorElement *hRes_S3RPhi = getter.get("DT/1DRecHits/Res/1D_S3RPhi_hRes");
  MonitorElement *hRes_S3RZ = getter.get("DT/1DRecHits/Res/1D_S3RZ_hRes");
  MonitorElement *hRes_S3RZ_W0 = getter.get("DT/1DRecHits/Res/1D_S3RZ_W0_hRes");
  MonitorElement *hRes_S3RZ_W1 = getter.get("DT/1DRecHits/Res/1D_S3RZ_W1_hRes");
  MonitorElement *hRes_S3RZ_W2 = getter.get("DT/1DRecHits/Res/1D_S3RZ_W2_hRes");

  MonitorElement *hPull_S3RPhi = getter.get("DT/1DRecHits/Pull/1D_S3RPhi_hPull");
  MonitorElement *hPull_S3RZ = getter.get("DT/1DRecHits/Pull/1D_S3RZ_hPull");
  MonitorElement *hPull_S3RZ_W0 = getter.get("DT/1DRecHits/Pull/1D_S3RZ_W0_hPull");
  MonitorElement *hPull_S3RZ_W1 = getter.get("DT/1DRecHits/Pull/1D_S3RZ_W1_hPull");
  MonitorElement *hPull_S3RZ_W2 = getter.get("DT/1DRecHits/Pull/1D_S3RZ_W2_hPull");

  Tutils util;
  util.drawGFit(hRes_S3RPhi->getTH1(), -0.2, 0.2, -0.1, 0.1);
  util.drawGFit(hRes_S3RZ->getTH1(), -0.2, 0.2, -0.1, 0.1);
  util.drawGFit(hRes_S3RZ_W0->getTH1(), -0.2, 0.2, -0.1, 0.1);
  util.drawGFit(hRes_S3RZ_W1->getTH1(), -0.2, 0.2, -0.1, 0.1);
  util.drawGFit(hRes_S3RZ_W2->getTH1(), -0.2, 0.2, -0.1, 0.1);

  util.drawGFit(hPull_S3RPhi->getTH1(), -5, 5, -5, 5);
  util.drawGFit(hPull_S3RZ->getTH1(), -5, 5, -5, 5);
  util.drawGFit(hPull_S3RZ_W0->getTH1(), -5, 5, -5, 5);
  util.drawGFit(hPull_S3RZ_W1->getTH1(), -5, 5, -5, 5);
  util.drawGFit(hPull_S3RZ_W2->getTH1(), -5, 5, -5, 5);

  if (doall_) {
    HEff1DHitHarvest hEff_S3RPhi("S3RPhi", booker, getter);
    HEff1DHitHarvest hEff_S3RZ("S3RZ", booker, getter);
    HEff1DHitHarvest hEff_S3RZ_W0("S3RZ_W0", booker, getter);
    HEff1DHitHarvest hEff_S3RZ_W1("S3RZ_W1", booker, getter);
    HEff1DHitHarvest hEff_S3RZ_W2("S3RZ_W2", booker, getter);

    if (doStep1_) {
      HEff1DHitHarvest hEff_S1RPhi("S1RPhi", booker, getter);
      HEff1DHitHarvest hEff_S1RZ("S1RZ", booker, getter);
      HEff1DHitHarvest hEff_S1RZ_W0("S1RZ_W0", booker, getter);
      HEff1DHitHarvest hEff_S1RZ_W1("S1RZ_W1", booker, getter);
      HEff1DHitHarvest hEff_S1RZ_W2("S1RZ_W2", booker, getter);
    }

    if (doStep2_) {
      HEff1DHitHarvest hEff_S2RPhi("S2RPhi", booker, getter);
      HEff1DHitHarvest hEff_S2RZ_W0("S2RZ_W0", booker, getter);
      HEff1DHitHarvest hEff_S2RZ_W1("S2RZ_W1", booker, getter);
      HEff1DHitHarvest hEff_S2RZ_W2("S2RZ_W2", booker, getter);
      HEff1DHitHarvest hEff_S2RZ("S2RZ", booker, getter);
    }
  }

  if (local_) {
    // Plots with finer granularity, not to be included in DQM
    TString name1 = "RPhi_W";
    TString name2 = "RZ_W";
    for (long w = 0; w <= 2; ++w) {
      for (long s = 1; s <= 4; ++s) {
        HEff1DHitHarvest hEff_S1RPhiWS(("S1" + name1 + w + "_St" + s).Data(), booker, getter);
        HEff1DHitHarvest hEff_S3RPhiWS(("S3" + name1 + w + "_St" + s).Data(), booker, getter);
        if (s != 4) {
          HEff1DHitHarvest hEff_S1RZWS(("S1" + name2 + w + "_St" + s).Data(), booker, getter);
          HEff1DHitHarvest hEff_S3RZWS(("S3" + name2 + w + "_St" + s).Data(), booker, getter);
        }
      }
    }
  }
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTRecHitClients);
