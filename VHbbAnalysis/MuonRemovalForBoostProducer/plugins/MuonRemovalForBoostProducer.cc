// -*- C++ -*-
//
// Package:    VHbbAnalysis/MuonRemovalForBoostProducer
// Class:      MuonRemovalForBoostProducer
// 
/**\class MuonRemovalForBoostProducer MuonRemovalForBoostProducer.cc VHbbAnalysis/MuonRemovalForBoostProducer/plugins/MuonRemovalForBoostProducer.cc

 Description: [one line class summary]
 
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Gregor Kasieczka (ETHZ) [gregor]
//         Created:  Thu, 28 Apr 2016 11:54:54 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <DataFormats/PatCandidates/interface/Muon.h>
#include <DataFormats/VertexReco/interface/Vertex.h>

//
// class declaration
//

class MuonRemovalForBoostProducer : public edm::stream::EDProducer<> {
   public:
      explicit MuonRemovalForBoostProducer(const edm::ParameterSet&);
      ~MuonRemovalForBoostProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      const edm::EDGetTokenT<std::vector<pat::Muon>> inputMuonToken_;        
      const edm::EDGetTokenT<std::vector<reco::Vertex>> inputVertexToken_;        

      // ----------member data ---------------------------
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
MuonRemovalForBoostProducer::MuonRemovalForBoostProducer(const edm::ParameterSet& iConfig) :
  inputMuonToken_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src"))),
  inputVertexToken_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vtx"))){

  // Output
  produces<std::vector<pat::Muon>>();      
}


MuonRemovalForBoostProducer::~MuonRemovalForBoostProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonRemovalForBoostProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   bool debug = false;

   // Get the muons
   Handle<std::vector<pat::Muon> >  input_mus;
   iEvent.getByToken(inputMuonToken_, input_mus);

   // Get the vertex 
   Handle<std::vector<reco::Vertex> >  input_vtxs;
   iEvent.getByToken(inputVertexToken_, input_vtxs);

   std::unique_ptr<std::vector<pat::Muon> > output_mus(new std::vector<pat::Muon>) ;
 
   // Selectrion criteria from 
   // https://twiki.cern.ch/twiki/bin/viewauth/CMS/TTbarHbbRun2ReferenceAnalysis_StartOf2016
   for (unsigned imu=0; imu != input_mus->size(); imu++){
   
     pat::Muon mu = input_mus->at(imu);
     
     // |eta| < 2.4
     if (! (fabs(mu.eta())<2.4)){
       if (debug)
	 std::cout << "Muon " << imu << " killed by eta" << std::endl;
       continue;
     }

     //  pT > 15
     if (! (mu.pt()>15)){
       if (debug)
	 std::cout << "Muon " << imu << " killed by pt" << std::endl;
       continue;
     }

     // Isolation
     if (! ((mu.pfIsolationR04().sumChargedHadronPt + std::max( mu.pfIsolationR04().sumNeutralHadronEt + mu.pfIsolationR04().sumPhotonEt - 0.5 * mu.pfIsolationR04().sumPUPt,0.0)) / mu.pt() < 0.25)){
       if (debug)
	 std::cout << "Muon " << imu << " killed by isolation" << std::endl;
       continue;
     }

     // Tight Muon ID
     if (! (mu.isTightMuon(input_vtxs->at(0)))){
       if (debug)
	 std::cout << "Muon " << imu << " killed by ID" << std::endl;
       continue;       
     }

     output_mus->push_back( input_mus->at(imu));
   }  
   
   if (debug)
     std::cout << input_mus->size() << " " << output_mus->size() << " muons for removal" << std::endl;
       
   iEvent.put(std::move(output_mus));
 
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
MuonRemovalForBoostProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
MuonRemovalForBoostProducer::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
MuonRemovalForBoostProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
MuonRemovalForBoostProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
MuonRemovalForBoostProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
MuonRemovalForBoostProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonRemovalForBoostProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonRemovalForBoostProducer);
