/****************************************************************************
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
****************************************************************************/

// TODO: clean
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "TFile.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionDiffPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSProtonReconstructionDiffPlotter(const edm::ParameterSet &);
  ~CTPPSProtonReconstructionDiffPlotter() override {}

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  void endJob() override;

  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsRef_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsTest_;

  std::string outputFile_;

  std::unique_ptr<TH1D> h_de_xi, h_de_th_x, h_de_th_y, h_de_vtx_y;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionDiffPlotter::CTPPSProtonReconstructionDiffPlotter(const edm::ParameterSet &ps)
    : tokenRecoProtonsRef_(consumes<reco::ForwardProtonCollection>(ps.getParameter<InputTag>("tagRecoProtonsRef"))),
      tokenRecoProtonsTest_(consumes<reco::ForwardProtonCollection>(ps.getParameter<InputTag>("tagRecoProtonsTest"))),

      outputFile_(ps.getParameter<string>("outputFile")),

      h_de_xi(new TH1D("", ";#Delta#xi", 200, -0.01, +0.01)),
      h_de_th_x(new TH1D("", ";#Delta#theta_{x}", 200, -100E-6, +100E-6)),
      h_de_th_y(new TH1D("", ";#Delta#theta_{y}", 200, -100E-6, +100E-6)),
      h_de_vtx_y(new TH1D("", ";#Deltay^{*}   (cm)", 200, -0.01, +0.01)) {}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionDiffPlotter::analyze(const edm::Event &event, const edm::EventSetup &iSetup) {
  // get input
  Handle<reco::ForwardProtonCollection> hRecoProtonsRef;
  event.getByToken(tokenRecoProtonsRef_, hRecoProtonsRef);

  Handle<reco::ForwardProtonCollection> hRecoProtonsTest;
  event.getByToken(tokenRecoProtonsTest_, hRecoProtonsTest);

  if (hRecoProtonsRef->size() != hRecoProtonsTest->size()) {
    edm::LogWarning("CTPPSProtonReconstructionDiffPlotter::analyze")
        << "Different number of Ref and Test protons. Skipping event.";
    return;
  }

  for (unsigned int i = 0; i < hRecoProtonsRef->size(); ++i) {
    const auto &pr_ref = hRecoProtonsRef->at(i);
    const auto &pr_test = hRecoProtonsTest->at(i);

    if (!pr_ref.validFit() || !pr_test.validFit())
      continue;

    h_de_xi->Fill(pr_test.xi() - pr_ref.xi());
    h_de_th_x->Fill(pr_test.thetaX() - pr_ref.thetaX());
    h_de_th_y->Fill(pr_test.thetaY() - pr_ref.thetaY());
    h_de_vtx_y->Fill(pr_test.vy() - pr_ref.vy());
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionDiffPlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  h_de_xi->Write("h_de_xi");
  h_de_th_x->Write("h_de_th_x");
  h_de_th_y->Write("h_de_th_y");
  h_de_vtx_y->Write("h_de_vtx_y");
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionDiffPlotter);
