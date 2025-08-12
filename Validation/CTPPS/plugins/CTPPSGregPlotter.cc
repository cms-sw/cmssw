// Implement and access Proton objects from cmssw/SimPPS/DirectSimProducer/plugins/PPSDirectProtonSimulation.cc
// Extract simple data like momentum from hepmc_prod

/****************************************************************************
* Authors:
*  Greg, based on Jan Kašpar
****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h" 
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h" 

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h" //Protons

//Beam
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h" 
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h" 

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "CondFormats/DataRecord/interface/PPSAssociationCutsRcd.h"
#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"


#include <cmath>
#include "TFile.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"


//----------------------------------------------------------------------------------------------------

class CTPPSGregPlotter : public edm::one::EDAnalyzer<> {
public:
    explicit CTPPSGregPlotter(const edm::ParameterSet &);
    ~CTPPSGregPlotter() override {}
 
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);


private:
    void analyze(const edm::Event &, const edm::EventSetup &) override;
    void endJob() override;
    
    std::string outputFile_;
    std::unique_ptr<TH1D> h_example;
    std::unique_ptr<TH1D> h_theta;
    std::unique_ptr<TH1D> h_phi;
    std::unique_ptr<TH1D> h_energy;
    std::unique_ptr<TH1D> h_pt;
    std::unique_ptr<TH1D> h_xi;


    edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_; //Protons
    edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_; //Beam

    bool debug = 0;
};

void CTPPSGregPlotter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("outputFile", "simu_2018_Greg.root")->setComment("name of the ouput file");
  desc.add<edm::InputTag>("hepMCTag", edm::InputTag("generator", "unsmeared"))->setComment("Input tag for HepMCProduct"); //Protons
  descriptions.add("ctppsGregPlotter", desc);
}

// Constructor definition 
CTPPSGregPlotter::CTPPSGregPlotter(const edm::ParameterSet &ps):
    outputFile_(ps.getParameter<std::string>("outputFile")),
    h_example (new TH1D("Example Histogram", "Energy", 2001, 3000., 7000.)),
    h_theta (new TH1D("Theta Degrees", "Theta", 1000, -0.025, 0.025)),
    h_phi (new TH1D("Phi", "Phi", 1000, -5., 5.)),
    h_energy (new TH1D("Energy", "Energy", 1800, 4800., 6600.)),
    h_pt (new TH1D("Pt", "Pt", 1000, -0.5, 4.5)),
    h_xi (new TH1D("Xi", "Xi", 350, -0.05, 0.3)),
    hepMCToken_(consumes<edm::HepMCProduct>(ps.getParameter<edm::InputTag>("hepMCTag"))), //Protons
    tokenBeamParameters_(esConsumes())


    {}


void CTPPSGregPlotter::analyze(const edm::Event &event, const edm::EventSetup &iSetup) {

  //Protons
  edm::Handle<edm::HepMCProduct> hepmc_prod; 
  event.getByToken(hepMCToken_, hepmc_prod); 
  
  //Beam
  auto const &beamParameters = iSetup.getData(tokenBeamParameters_);
  if (hepmc_prod.isValid()) {
    auto evt = hepmc_prod->GetEvent();
    if (evt) {
      // Loop over generated particles in the HepMC event
      for (auto it_part = evt->particles_begin(); it_part != evt->particles_end(); ++it_part) {
        const HepMC::GenParticle* part = *it_part;

        // Extract momentum information from the GenParticle
        const HepMC::FourVector& momentum = part->momentum();
        if (part->pdg_id() != 2212) continue;
        if(debug){
          std::cout << "  Particle PDG ID: " << part->pdg_id()
                    << ", Status: " << part->status()
                    << ", Px: " << momentum.px()
                    << ", Py: " << momentum.py()
                    << ", Pz: " << momentum.pz()
                    << ", Energy: " << momentum.e()
                    << std::endl;

          std::cout << beamParameters.getBeamMom45() << std::endl;
        }

        h_example->Fill(momentum.e());
        // Theta Transformation
        double theta_deg = 200;
        theta_deg = momentum.theta() / M_PI * 180;
        if (theta_deg > 90) theta_deg -= 180; 

        if (theta_deg != 200) h_theta->Fill(theta_deg);
        h_phi->Fill(momentum.phi());
        h_energy->Fill(momentum.e());
        h_pt->Fill(sqrt(pow(momentum.px(),2)+ pow(momentum.py(),2)));
        //from PPSDirectProtonSimulation.cc
        h_xi->Fill(1.- momentum.rho()/ beamParameters.getBeamMom56());
      }
    } else {
      std::cerr << "Error: HepMC::GenEvent is null." << std::endl;
    }
  } else {
    std::cerr << "Error: HepMCProduct is not valid." << std::endl;
  }


}

//----------------------------------------------------------------------------------------------------


void CTPPSGregPlotter::endJob() {

  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  h_example->Write();
  h_theta->Write();
  h_phi->Write();
  h_energy->Write();
  h_pt->Write();
  h_xi->Write();

  std::cout << "GregPlotter worked" << std::endl;

}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSGregPlotter);