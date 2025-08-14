/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Greg
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h" //Fix for not recognising the header

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/PPSDirectSimulationDataRcd.h"
#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "CondFormats/DataRecord/interface/PPSPixelTopologyRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <unordered_map>

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TF1.h"
#include "TF2.h"
#include "TFile.h"
#include "CLHEP/Random/RandFlat.h"

#include <map>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSGregDucer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSGregDucer(const edm::ParameterSet &);
  ~CTPPSGregDucer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:

  void produce(edm::Event &, const edm::EventSetup &) override;

  void getCut();

  bool applyCut(const HepMC::GenParticle *part,
                const CTPPSBeamParameters &beamParameters) const;



  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;

  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_; 

  bool debug = 0;
  std::string filename1_;
  std::map<double, std::map<double, std::pair<double, double>>> cutMap1_;
};
//----------------------------------------------------------------------------------------------------

CTPPSGregDucer::CTPPSGregDucer(const edm::ParameterSet &ps):
  tokenTracks_(consumes<CTPPSLocalTrackLiteCollection>(ps.getParameter<edm::InputTag>("tagTracks"))),
  hepMCToken_(consumes<edm::HepMCProduct>(ps.getParameter<edm::InputTag>("hepMCTag"))), //Protons
  tokenBeamParameters_(esConsumes())
  {
    produces<CTPPSLocalTrackLiteCollection>(); 
    produces<edm::HepMCProduct>("selectedProtons");
    filename1_ = ps.getParameter<std::string>("filename");
    getCut();
  }


//----------------------------------------------------------------------------------------------------

void CTPPSGregDucer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tagTracks", edm::InputTag("ctppsLocalTrackLiteProducer"))->setComment("Input tag for CTPPSLocalTrackLiteCollection");
  desc.add<edm::InputTag>("hepMCTag", edm::InputTag("generator", "unsmeared"))->setComment("Input tag for HepMCProduct"); //Protons
  desc.add<std::string>("filename", "default_file.txt")->setComment("File with cut definitions"); 
  descriptions.add("ctppsGregDucer", desc);

}

//----------------------------------------------------------------------------------------------------


void CTPPSGregDucer::produce(edm::Event &event, const edm::EventSetup &iSetup) {
  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  event.getByToken(tokenTracks_, hTracks);

  //Protons
  edm::Handle<edm::HepMCProduct> hepmc_prod; 
  event.getByToken(hepMCToken_, hepmc_prod); 
  
  //Beam
  auto const &beamParameters = iSetup.getData(tokenBeamParameters_);


  // Reference cutMap1


  // loop over event vertices
  auto evt = hepmc_prod->GetEvent();
  auto filteredEvent = std::make_unique<HepMC::GenEvent>(evt->signal_process_id(), evt->event_number());

  for (auto it_vtx = evt->vertices_begin(); it_vtx != evt->vertices_end(); ++it_vtx) {
  auto vtx = new HepMC::GenVertex((*it_vtx)->position());

  for (auto it_part = (*it_vtx)->particles_out_const_begin(); it_part != (*it_vtx)->particles_out_const_end(); ++it_part) {
    auto part = *it_part;
    if (applyCut(part, beamParameters)) {
      auto newPart = new HepMC::GenParticle(*part); // clone
      vtx->add_particle_out(newPart);
    }
  }

    if (vtx->particles_out_size() > 0) {
      filteredEvent->add_vertex(vtx);
    } else {
      delete vtx;
      }
  }

  auto output = std::make_unique<edm::HepMCProduct>();
  output->addHepMCData(filteredEvent.release()); // HepMCProduct takes ownership
  event.put(std::move(output), "selectedProtons");
}
//----------------------------------------------------------------------------------------------------


  void CTPPSGregDucer::getCut() {
    std::ifstream inputFile1(filename1_);

    if (!inputFile1.is_open()) {
        throw cms::Exception("IOError") << "Could not open file " << filename1_;
    }

    // 1st value specifies no of phi-s
    unsigned int n_phis = 0;
    inputFile1 >> n_phis;

    if(debug) std::cout << n_phis << std::endl; //Works

    std::vector<double> phis(n_phis);
    double temp_phi;

    for (unsigned int i = 0; i < n_phis; ++i) {
        inputFile1 >> phis[i] >> temp_phi; // Read the pair, store the first one
        if(debug) std::cout << temp_phi << std::endl;

    }



    double xi, theta_min, theta_max;

    while (inputFile1 >> xi) {
      for (unsigned int i = 0; i < n_phis; ++i) {
            inputFile1 >> theta_min >> theta_max;
            if(debug) std::cout << xi << " " << theta_min << " " << theta_max << std::endl;
            cutMap1_[xi][phis[i]] = {theta_min, theta_max};
        }
    }

    inputFile1.close();
   
  }


//----------------------------------------------------------------------------------------------------

bool CTPPSGregDucer::applyCut(const HepMC::GenParticle *part,
                              const CTPPSBeamParameters &beamParameters) const {
    if(debug) if (part->pdg_id() != 2212) std::cout << "non proton" << std::endl;
    // double xi = 0;
    // xi = 1.- part->momentum().rho()/ beamParameters.getBeamMom56();
    double xi_cutTest, phi_cutTest;
    xi_cutTest =  0.039000;
    phi_cutTest = 2.356194;
    std::pair<double, double> thetas_cutTest = cutMap1_.at(xi_cutTest).at(phi_cutTest);   
    double theta_min = thetas_cutTest.first;
    double theta_max = thetas_cutTest.second;


    // accept only stable protons
    if (part->pdg_id() == 2212){
      if (part->status() != 1 && part->status() < 83) return false;
      // if (part->momentum().perp()< 0.5) return false;
      // if (xi < 0.1) return false;
      if(part->momentum().theta() > theta_max || part->momentum().theta() < theta_min ) return false; 
      
    }

    return true;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSGregDucer);
