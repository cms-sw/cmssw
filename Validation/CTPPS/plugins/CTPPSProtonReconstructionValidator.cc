/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "TFile.h"
#include "TH1D.h"

#include <map>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionValidator : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSProtonReconstructionValidator(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtons_;
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> opticsESToken_;
  double chiSqCut_;
  std::string outputFile_;

  struct RPPlots {
    std::unique_ptr<TH1D> h_de_x, h_de_y;
    RPPlots()
        : h_de_x(new TH1D("", ";#Deltax   (mm)", 1000, -1., +1.)),
          h_de_y(new TH1D("", ";#Deltay   (mm)", 1000, -1., +1.)) {}

    void fill(double de_x, double de_y) {
      h_de_x->Fill(de_x);
      h_de_y->Fill(de_y);
    }

    void write() const {
      h_de_x->Write("h_de_x");
      h_de_y->Write("h_de_y");
    }
  };

  std::map<unsigned int, RPPlots> rp_plots_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionValidator::CTPPSProtonReconstructionValidator(const edm::ParameterSet& iConfig)
    : tokenRecoProtons_(consumes<reco::ForwardProtonCollection>(iConfig.getParameter<edm::InputTag>("tagRecoProtons"))),
      opticsESToken_(esConsumes()),
      chiSqCut_(iConfig.getParameter<double>("chiSqCut")),
      outputFile_(iConfig.getParameter<string>("outputFile")) {}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get conditions
  const auto& opticalFunctions = iSetup.getData(opticsESToken_);

  // stop if conditions invalid
  if (opticalFunctions.empty())
    return;

  // get input
  Handle<reco::ForwardProtonCollection> hRecoProtons;
  iEvent.getByToken(tokenRecoProtons_, hRecoProtons);

  // process tracks
  for (const auto& pr : *hRecoProtons) {
    if (!pr.validFit())
      continue;

    if (pr.chi2() > chiSqCut_)
      continue;

    for (const auto& tr : pr.contributingLocalTracks()) {
      CTPPSDetId rpId(tr->rpId());
      unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

      // skip other than tracking RPs
      if (rpId.subdetId() != CTPPSDetId::sdTrackingStrip && rpId.subdetId() != CTPPSDetId::sdTrackingPixel)
        continue;

      // try to get optics for the RP
      if (opticalFunctions.count(rpId) == 0)
        continue;
      const auto& func = opticalFunctions.at(rpId);

      // do propagation
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_beam = {0., 0., 0., 0., 0.};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_beam;
      func.transport(k_in_beam, k_out_beam);

      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in = {
          -pr.vx(), -pr.thetaX(), pr.vy(), pr.thetaY(), pr.xi()};  // conversions: CMS --> LHC convention
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out;
      func.transport(k_in, k_out);

      // fill plots
      const double de_x = (k_out.x - k_out_beam.x) * 10. - tr->x();  // conversions: cm --> mm
      const double de_y = (k_out.y - k_out_beam.y) * 10. - tr->y();  // conversions: cm --> mm

      rp_plots_[rpDecId].fill(de_x, de_y);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidator::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto& p : rp_plots_) {
    gDirectory = f_out->mkdir(Form("%u", p.first));
    p.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionValidator);
