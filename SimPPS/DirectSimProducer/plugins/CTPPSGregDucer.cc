/****************************************************************************
 * Authors:
 *   Grzegorz Jędrzejowski,
 *    based on Jan Kašpar.
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

  void interpolate(double xi,
                   double phi);

  double interpolate_step(double exp_value,
                          double low_bound,
                          double up_bound,
                          double theta_l,
                          double theta_u) const;

  bool applyCut(const HepMC::GenParticle *part,
                const CTPPSBeamParameters &beamParameters);



  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;

  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_; 

  bool debug = 0;
  std::string filename1_;
  std::map<double, std::map<double, std::pair<double, double>>> cutMap1_;
  double theta_min = 0;
  double theta_max = 0;
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


    std::vector<double> phis(n_phis);
    double temp_phi;

    for (unsigned int i = 0; i < n_phis; ++i) {
        inputFile1 >> phis[i] >> temp_phi; // Read the pair, store the first one

    }



    double xi, theta_min, theta_max;

    while (inputFile1 >> xi) {
      for (unsigned int i = 0; i < n_phis; ++i) {
            inputFile1 >> theta_min >> theta_max;
            cutMap1_[xi][phis[i]] = {theta_min, theta_max};
        }
    }

    inputFile1.close();
   
  }


//----------------------------------------------------------------------------------------------------

bool CTPPSGregDucer::applyCut(const HepMC::GenParticle *part,
                              const CTPPSBeamParameters &beamParameters) {
    if(debug) if (part->pdg_id() != 2212) std::cout << "non proton" << std::endl;

    double xi, phi;
    xi = 1.- part->momentum().rho()/ beamParameters.getBeamMom56();
    phi = part->momentum().phi();
    interpolate(xi, phi);

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
void CTPPSGregDucer::interpolate(double xi,
                                 double phi){


    //transforming phi to (0;2pi) convention

    if(phi < 0) {
      phi += 2.0 * M_PI;
    }


    auto it_xi_upper = cutMap1_.upper_bound(xi);
    double xi1, xi2;

    if (it_xi_upper == cutMap1_.end()) {
    // xi is greater than all keys in the map.
    // Set both xi1 and xi2 to the largest key.
    auto it_xi_lower = std::prev(it_xi_upper);
    xi1 = it_xi_lower->first;
    xi2 = it_xi_lower->first;
    } else if (it_xi_upper == cutMap1_.begin()) {
        // xi is less than all keys in the map.
        // Set both xi1 and xi2 to the smallest key.
        xi1 = it_xi_upper->first;
        xi2 = it_xi_upper->first;
    } else {
        // xi is either a key or is between two keys.
        // The key before it_xi_upper is the smaller one.
        auto it_xi_lower = std::prev(it_xi_upper);
        xi1 = it_xi_lower->first;
        xi2 = it_xi_upper->first;
    }


    auto& innerMap = cutMap1_[xi1];
    auto it_phi_upper = innerMap.upper_bound(phi);
    double phi1, phi2;

   if (it_phi_upper == innerMap.begin()) {
    // phi is less than all keys in the map.
    // Set both phi1 and phi2 to the smallest key.
    phi1 = it_phi_upper->first;
    phi2 = it_phi_upper->first;
    } else if (it_phi_upper == innerMap.end()) {
        // phi is greater than all keys in the map.
        // Set both phi1 and phi2 to the largest key.
        auto it_phi_lower = std::prev(it_phi_upper);
        phi1 = it_phi_lower->first;
        phi2 = it_phi_lower->first;
    } else {
        // phi is either a key or is between two keys.
        // The key before it_phi_upper is the smaller one.
        auto it_phi_lower = std::prev(it_phi_upper);
        phi1 = it_phi_lower->first;
        phi2 = it_phi_upper->first;
    }


    // Access the pair for (xi1, phi1) and assign its components
    std::pair<double, double> thetaPair11 = cutMap1_[xi1][phi1];
    double theta_m11 = thetaPair11.first;
    double theta_p11 = thetaPair11.second;

    // Access the pair for (xi1, phi2) and assign its components
    std::pair<double, double> thetaPair12 = cutMap1_[xi1][phi2];
    double theta_m12 = thetaPair12.first;
    double theta_p12 = thetaPair12.second;

    // Access the pair for (xi2, phi1) and assign its components
    std::pair<double, double> thetaPair21 = cutMap1_[xi2][phi1];
    double theta_m21 = thetaPair21.first;
    double theta_p21 = thetaPair21.second;

    // Access the pair for (xi2, phi2) and assign its components
    std::pair<double, double> thetaPair22 = cutMap1_[xi2][phi2];
    double theta_m22 = thetaPair22.first;
    double theta_p22 = thetaPair22.second;

    double theta_interm_m1, theta_interm_p1, theta_interm_m2, theta_interm_p2;

    // xi1 
    theta_interm_m1 = interpolate_step(phi, phi1, phi2, theta_m11, theta_m12);
    theta_interm_p1 = interpolate_step(phi, phi1, phi2, theta_p11, theta_p12);

    // xi2
    theta_interm_m2 = interpolate_step(phi, phi1, phi2, theta_m21, theta_m22);
    theta_interm_p2 = interpolate_step(phi, phi1, phi2, theta_p21, theta_p22);

    // Result
    theta_min = interpolate_step(xi, xi1, xi2, theta_interm_m1, theta_interm_m2);
    theta_max = interpolate_step(xi, xi1, xi2, theta_interm_p1, theta_interm_p2);

    if(debug) std::cout << "Xi: " << xi << " Phi:" << phi << " Theta_min thata_max: " << theta_min << " " << theta_max << std::endl;

}


//----------------------------------------------------------------------------------------------------

  double CTPPSGregDucer::interpolate_step(double exp_value,
                          double low_bound,
                          double up_bound,
                          double theta_l,
                          double theta_u) const{
    double theta = 0;

    theta = (exp_value - low_bound) * (theta_u - theta_l) / (up_bound - low_bound) + theta_l;

    return theta;                     
  }

DEFINE_FWK_MODULE(CTPPSGregDucer);
