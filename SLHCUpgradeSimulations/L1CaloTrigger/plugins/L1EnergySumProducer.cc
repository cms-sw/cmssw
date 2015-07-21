// -*- C++ -*-
//
// Package:    L1CaloTrigger
// Class:      L1EnergySumProducer
// 
/**\class L1EnergySumProducer L1EnergySumProducer.cc SLHCUpgradeSimulations/L1CaloTrigger/plugins/L1EnergySumProducer.cc

 Description: Produces energy sums from L1 calorimeter objects

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mark Baber
//         Created:  Tue, 18 Mar 2014 20:07:42 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TLorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include <iostream>


#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"



//
// class declaration
//

class L1EnergySumProducer : public edm::EDProducer {
   public:
      explicit L1EnergySumProducer(const edm::ParameterSet&);
      ~L1EnergySumProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      edm::ParameterSet conf_;

      // Tower geometry converter 
      TriggerTowerGeometry mTowerGeo;


};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1EnergySumProducer::L1EnergySumProducer(const edm::ParameterSet& iConfig): conf_(iConfig)
{

  produces<l1extra::L1EtMissParticleCollection>( "MET" );


}


L1EnergySumProducer::~L1EnergySumProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1EnergySumProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<l1extra::L1EtMissParticleCollection> outputMET(new l1extra::L1EtMissParticleCollection());


  // TT collection 
  edm::Handle<l1slhc::L1CaloTowerCollection> caloTowers;
  iEvent.getByLabel(conf_.getParameter<edm::InputTag>("CalorimeterTowers"), caloTowers);
  if(!caloTowers.isValid()){
    edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("CalorimeterTowers") << std::endl;
  }



  math::PtEtaPhiMLorentzVector MET, protoMET;
  double ET = 0;

  // Calculate TT-energy sums
  for( l1slhc::L1CaloTowerCollection::const_iterator lTT_It = caloTowers->begin();
       lTT_It != caloTowers->end() ; ++lTT_It ){

    // Energies in GeV
    double E      = 0.5*lTT_It->E();
    double H      = 0.5*lTT_It->H();
    int iEta      = lTT_It->iEta();
    int iPhi      = lTT_It->iPhi();

    //    double Eta = mTowerGeo.eta(iEta);
    double Phi = mTowerGeo.phi(iPhi);


    // Restrict to central TTs                                                                                                              
    if (abs(iEta) > 28)
      continue;


    // Calculate MET
    protoMET.SetCoordinates( E + H, 0, Phi, 0 );
    MET += protoMET;
    ET  += E + H;


  }


  
  l1extra::L1EtMissParticle l1extraMET( MET, l1extra::L1EtMissParticle::kMET, ET,
					edm::Ref< L1GctEtMissCollection >(), edm::Ref< L1GctEtTotalCollection >(),
					edm::Ref< L1GctHtMissCollection >(), edm::Ref< L1GctEtHadCollection >()  ,  0);
  
  outputMET->push_back( l1extraMET );
  

  
  iEvent.put( outputMET, "MET" );
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1EnergySumProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1EnergySumProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1EnergySumProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1EnergySumProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1EnergySumProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1EnergySumProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1EnergySumProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1EnergySumProducer);
