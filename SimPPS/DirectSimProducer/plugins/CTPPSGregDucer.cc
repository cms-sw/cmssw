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

//----------------------------------------------------------------------------------------------------

class CTPPSGregDucer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSGregDucer(const edm::ParameterSet &);
  ~CTPPSGregDucer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  bool applyCut(const HepMC::GenParticle *part) const;


  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;

  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_; 


};
//----------------------------------------------------------------------------------------------------

CTPPSGregDucer::CTPPSGregDucer(const edm::ParameterSet &ps):
  tokenTracks_(consumes<CTPPSLocalTrackLiteCollection>(ps.getParameter<edm::InputTag>("tagTracks"))),
  hepMCToken_(consumes<edm::HepMCProduct>(ps.getParameter<edm::InputTag>("hepMCTag"))), //Protons
  tokenBeamParameters_(esConsumes())
  {
    produces<CTPPSLocalTrackLiteCollection>(); 
    produces<edm::HepMCProduct>("selectedProtons");
  }


//----------------------------------------------------------------------------------------------------

void CTPPSGregDucer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tagTracks", edm::InputTag("ctppsLocalTrackLiteProducer"))->setComment("Input tag for CTPPSLocalTrackLiteCollection");
  desc.add<edm::InputTag>("hepMCTag", edm::InputTag("generator", "unsmeared"))->setComment("Input tag for HepMCProduct"); //Protons
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
  // auto const &beamParameters = iSetup.getData(tokenBeamParameters_);

  // loop over event vertices
  auto evt = hepmc_prod->GetEvent();
  auto filteredEvent = std::make_unique<HepMC::GenEvent>(evt->signal_process_id(), evt->event_number());

  for (auto it_vtx = evt->vertices_begin(); it_vtx != evt->vertices_end(); ++it_vtx) {
  auto vtx = new HepMC::GenVertex((*it_vtx)->position());

  for (auto it_part = (*it_vtx)->particles_out_const_begin(); it_part != (*it_vtx)->particles_out_const_end(); ++it_part) {
    auto part = *it_part;
    if (applyCut(part)) {
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

bool CTPPSGregDucer::applyCut(const HepMC::GenParticle *part) const {
    // std::cout << "Greg Produced" << std::endl; //It works
    // accept only stable protons
    if (part->pdg_id() != 2212) return false;
    if (part->status() != 1 && part->status() < 83) return false;
    if (part->momentum().perp()< 0.5) return false;

    return true;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSGregDucer);
