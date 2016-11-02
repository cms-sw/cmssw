// -*- C++ -*-
//
// Package:    VHbbAnalysis/ElectronRemovalForBoostProducer
// Class:      ElectronRemovalForBoostProducer
// 
/**\class ElectronRemovalForBoostProducer ElectronRemovalForBoostProducer.cc VHbbAnalysis/ElectronRemovalForBoostProducer/plugins/ElectronRemovalForBoostProducer.cc
 
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

#include <DataFormats/PatCandidates/interface/Electron.h>

float effArea(float aeta){

  if (aeta < 1.000)
    return 0.1752;
  if (aeta < 1.479)
    return 0.1862;
  if (aeta < 2.000) 
    return 0.1411; 
  if (aeta < 2.200) 
    return 0.1534; 
  if (aeta < 2.300) 
    return 0.1903; 
  if (aeta < 2.400) 
    return 0.2243;
   
  return 0.2687;
}


//
// class declaration
//

class ElectronRemovalForBoostProducer : public edm::stream::EDProducer<> {
   public:
      explicit ElectronRemovalForBoostProducer(const edm::ParameterSet&);
      ~ElectronRemovalForBoostProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      const edm::EDGetTokenT<std::vector<pat::Electron>> inputElectronToken_;        
      const edm::EDGetTokenT<edm::ValueMap<bool>> inputmvaIDMapToken_;        
      const edm::EDGetTokenT<double> inputRhoToken_;        



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
ElectronRemovalForBoostProducer::ElectronRemovalForBoostProducer(const edm::ParameterSet& iConfig) :
  inputElectronToken_(consumes<std::vector<pat::Electron>>(iConfig.getParameter<edm::InputTag>("src"))),
  inputmvaIDMapToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("mvaIDMap"))),
  inputRhoToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))) {
  // Output
  produces<std::vector<pat::Electron>>();      
}


ElectronRemovalForBoostProducer::~ElectronRemovalForBoostProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronRemovalForBoostProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // Get the electrons
   Handle<std::vector<pat::Electron> >  input_es;
   iEvent.getByToken(inputElectronToken_, input_es);

   // Get rho
   Handle<double > rho;
   iEvent.getByToken(inputRhoToken_, rho);

   // Get MVA ID Map
   Handle<edm::ValueMap<bool> > mvaIDMap;
   iEvent.getByToken(inputmvaIDMapToken_, mvaIDMap);

   std::unique_ptr<std::vector<pat::Electron> > output_es(new std::vector<pat::Electron>) ;
 
   // Selectrion criteria from https://twiki.cern.ch/twiki/bin/viewauth/CMS/TTbarHbbRun2ReferenceAnalysis_StartOf2016
   for (unsigned ie=0; ie != input_es->size(); ie++){
   
     pat::Electron e = input_es->at(ie);
     
     // |eta| < 2.4
     if (! (fabs(e.eta())<2.4)){
       //std::cout << ie << " killed by eta" << std::endl;
       continue;
     }

     //  pT > 15
     if (! ((e.pt())>15)){
       //std::cout << ie << " killed by pt " << e.pt() << std::endl;
       continue;
     }

     // Additional selection
     if (! (((fabs(e.superCluster()->eta()) < 1.4442 && 
	      e.full5x5_sigmaIetaIeta() < 0.012 && 
	      e.hcalOverEcal() < 0.09 && 
	      (e.ecalPFClusterIso() / e.pt()) < 0.37 && 
	      (e.hcalPFClusterIso() / e.pt()) < 0.25 && 
	      (e.dr03TkSumPt() / e.pt()) < 0.18 && 
	      fabs(e.deltaEtaSuperClusterTrackAtVtx()) < 0.0095 && 
	      fabs(e.deltaPhiSuperClusterTrackAtVtx()) < 0.065 ) || 
	   (fabs(e.superCluster()->eta()) > 1.5660 && 
	      e.full5x5_sigmaIetaIeta() < 0.033 && 
	      e.hcalOverEcal() <0.09 && 
	      (e.ecalPFClusterIso() / e.pt()) < 0.45 && 
	      (e.hcalPFClusterIso() / e.pt()) < 0.28 && 
	    (e.dr03TkSumPt() / e.pt()) < 0.18 )))){
       //std::cout << ie << " killed by extra selection" << std::endl;
       continue;
     }


     // Electron ID       
     if (! (mvaIDMap->get(ie))){
       //std::cout << ie << " killed by ID " << ((mvaIDMap->get(ie))) << std::endl;
       continue;
     }


     // Isolation
     if (! (((e.pfIsolationVariables().sumChargedHadronPt + std::max(0.0,e.pfIsolationVariables().sumNeutralHadronEt + e.pfIsolationVariables().sumPhotonEt - (*rho)*effArea(fabs(e.superCluster()->eta()))))/e.pt()) < 0.15)) {
       //std::cout << ie << " killed by iso" << std::endl;
       continue;
     }
     output_es->push_back( input_es->at(ie));
   }  
   
   //std::cout << "Electrons: " << output_es->size() << " / " << input_es->size() << std::endl;
       
   iEvent.put(std::move(output_es));
 
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
ElectronRemovalForBoostProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
ElectronRemovalForBoostProducer::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
ElectronRemovalForBoostProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
ElectronRemovalForBoostProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
ElectronRemovalForBoostProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
ElectronRemovalForBoostProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ElectronRemovalForBoostProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronRemovalForBoostProducer);
