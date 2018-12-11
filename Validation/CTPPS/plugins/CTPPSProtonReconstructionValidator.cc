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

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsCollection.h"

#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include "TFile.h"
#include "TH1D.h"

#include <map>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionValidator : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSProtonReconstructionValidator(const edm::ParameterSet&);

    ~CTPPSProtonReconstructionValidator() {}

  private:
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    virtual void endJob() override;

    edm::EDGetTokenT<std::vector<reco::ProtonTrack>> tokenRecoProtons_;

    double chiSqCut_;

    std::string outputFile_;

    edm::ESWatcher<LHCInfoRcd> lhcInfoWatcher_;
    float currentCrossingAngle_;

    std::unordered_map<unsigned int, LHCOpticalFunctionsSet> opticalFunctions_;

    struct RPPlots
    {
      TH1D *h_de_x = NULL, *h_de_y;

      void init()
      {
        h_de_x = new TH1D("", ";#Deltax   (mm)", 100, -2E-3, +2E-3);
        h_de_y = new TH1D("", ";#Deltay   (mm)", 100, -2E-3, +2E-3);
      }

      void fill(double de_x, double de_y)
      {
        if (h_de_x == NULL)
          init();

        h_de_x->Fill(de_x);
        h_de_y->Fill(de_y);
      }

      void write() const
      {
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

CTPPSProtonReconstructionValidator::CTPPSProtonReconstructionValidator(const edm::ParameterSet& iConfig) :
  tokenRecoProtons_( consumes<std::vector<reco::ProtonTrack>>(iConfig.getParameter<edm::InputTag>("tagRecoProtons")) ),
  chiSqCut_( iConfig.getParameter<double>("chiSqCut") ),
  outputFile_(iConfig.getParameter<string>("outputFile"))
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidator::analyze(const edm::Event& iEvent, const edm::EventSetup &iSetup)
{
  // get conditions
  edm::ESHandle<LHCInfo> hLHCInfo;
  iSetup.get<LHCInfoRcd>().get(hLHCInfo);

  edm::ESHandle<LHCOpticalFunctionsCollection> hOpticalFunctionCollection;
  iSetup.get<CTPPSOpticsRcd>().get(hOpticalFunctionCollection);

  // interpolate optical functions, if needed
  if (lhcInfoWatcher_.check(iSetup))
  {
    const LHCInfo* pLHCInfo = hLHCInfo.product();
    if (pLHCInfo->crossingAngle() != currentCrossingAngle_)
    {
      currentCrossingAngle_ = pLHCInfo->crossingAngle();

      opticalFunctions_.clear();
      hOpticalFunctionCollection->interpolateFunctions(currentCrossingAngle_, opticalFunctions_);
      for (auto &p : opticalFunctions_)
        p.second.initializeSplines();
    }
  } 

  // stop if conditions invalid
  if (currentCrossingAngle_ <= 0.)
    return;

  // get input
  Handle<vector<reco::ProtonTrack>> hRecoProtons;
  iEvent.getByToken(tokenRecoProtons_, hRecoProtons);

  // process tracks
  for (const auto &pr : *hRecoProtons)
  {
    if (! pr.validFit())
      continue;

    if (pr.chi2() > chiSqCut_)
      continue;

    for (const auto &tr : pr.contributingLocalTracks())
    {
      CTPPSDetId rpId(tr->getRPId());
      unsigned int rpDecId = rpId.arm()*100 + rpId.station()*10 + rpId.rp();

      auto it = opticalFunctions_.find(rpId);

      LHCOpticalFunctionsSet::Kinematics k_in_beam = { 0., 0., 0., 0., 0. };
      LHCOpticalFunctionsSet::Kinematics k_out_beam; 
      it->second.transport(k_in_beam, k_out_beam);

      LHCOpticalFunctionsSet::Kinematics k_in = { pr.vx() * 1E-2, -pr.thetaX(), pr.vy() * 1E-2, pr.thetaY(), pr.xi() };  // conversions: cm --> m, CMS --> LHC convention
      LHCOpticalFunctionsSet::Kinematics k_out; 
      it->second.transport(k_in, k_out);

      const double de_x = (k_out.x - k_out_beam.x) * 1E3 - tr->getX();  // conversions: m --> mm
      const double de_y = (k_out.y - k_out_beam.y) * 1E3 - tr->getY();  // conversions: m --> mm

      rp_plots_[rpDecId].fill(de_x, de_y);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionValidator::endJob()
{
  TFile *f_out = TFile::Open(outputFile_.c_str(), "recreate");

  for (const auto &p : rp_plots_)
  {
    char buf[100];
    sprintf(buf, "%u", p.first);
    gDirectory = f_out->mkdir(buf);
    p.second.write();
  }

  delete f_out;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionValidator);
