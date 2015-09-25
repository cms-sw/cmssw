// -*- C++ -*-
//
// Package:    DYGenFilter
// Class:      DYGenFilter
// 
/**\class DYGenFilter DYGenFilter.cc MyAna/DYGenFilter/src/DYGenFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Mon Dec  5 16:42:11 CET 2011
// $Id: DYGenFilter.cc,v 1.2 2012/04/24 10:07:09 fruboes Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h" 

//
// class declaration
//

class DYGenFilter : public edm::EDFilter {
   public:
      explicit DYGenFilter(const edm::ParameterSet&);
      ~DYGenFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual bool beginRun(edm::Run&, edm::EventSetup const&);
      virtual bool endRun(edm::Run&, edm::EventSetup const&);
      virtual bool beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual bool endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

  
      // ----------member data ---------------------------
      int pdgCode_;
      double etaMax_;
      double ptMin_;
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
DYGenFilter::DYGenFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  pdgCode_ = abs(iConfig.getUntrackedParameter<int>("code"));
  etaMax_ = iConfig.getUntrackedParameter<double>("etaMax");
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin");

}


DYGenFilter::~DYGenFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
DYGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  Handle< HepMCProduct > genh;
  iEvent.getByLabel("generatorSmeared", genh);



  HepMC::GenEvent::particle_const_iterator it, itE;
  it = genh->GetEvent()->particles_begin();
  itE = genh->GetEvent()->particles_end();

  float etaMax_ = 2.4;

  //std::cout << "-------------" << std::endl;
  std::vector<float> pts(2,0);
  for(;it!=itE;++it){
    int pdg = abs( (*it)->pdg_id() ) ;
    if (pdg != pdgCode_) continue;
    if ( abs((*it)->momentum().eta() )>etaMax_) continue;
    if ( (*it)->momentum().perp() < ptMin_ ) continue;
    int charge = pdg/(*it)->pdg_id();
    float pt = (*it)->momentum().perp();
    //std::cout << charge << " " << pt << std::endl;
    int index = 0;
    if (charge > 0) index = 1;

    if ( pts.at(index)<pt) pts.at(index) = pt;

  }
 
  bool ret = true;
  if ( pts[0]<ptMin_ || pts[1]<ptMin_) ret =false;
  //std::cout << "XXX " << pts[0] << " " << pts[1] <<  " " << ret << std::endl;
  return ret;
}

// ------------ method called once each job just before starting event loop  ------------
void 
DYGenFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DYGenFilter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
bool 
DYGenFilter::beginRun(edm::Run&, edm::EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
DYGenFilter::endRun(edm::Run&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
DYGenFilter::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
DYGenFilter::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DYGenFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(DYGenFilter);
