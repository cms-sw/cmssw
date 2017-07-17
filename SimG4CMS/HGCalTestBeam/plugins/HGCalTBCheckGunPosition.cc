// -*- C++ -*-
//
// Package:    HGCalTestBeam/HGCalTBCheckGunPostion
// Class:      HGCalTBCheckGunPostion
// 
/**\class HGCalTBCheckGunPostion HGCalTBCheckGunPostion.cc Geometry/HGCalTestBeam/plugins/HGCalTBCheckGunPostion.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Shilpi Jain
//         Created:  Wed, 31 Aug 2016 17:47:22 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>


//
// class declaration
//

class HGCalTBCheckGunPostion : public edm::stream::EDFilter<> {

public:
  explicit HGCalTBCheckGunPostion(const edm::ParameterSet&);
  ~HGCalTBCheckGunPostion();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginStream(edm::StreamID) override {}
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override {}

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct>      hepMCproductLabel_;
  bool                                     verbosity_, method2_;
  double                                   tan30deg_, hexwidth_, hexside_;
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
HGCalTBCheckGunPostion::HGCalTBCheckGunPostion(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  edm::InputTag tmp0 = iConfig.getParameter<edm::InputTag>("HepMCProductLabel");
  verbosity_         = iConfig.getUntrackedParameter<bool>("Verbosity",false);
  method2_           = iConfig.getUntrackedParameter<bool>("Method2",false);
  hepMCproductLabel_ = consumes<edm::HepMCProduct>(tmp0);
  
  //hexside = 7; //cm - check it
  tan30deg_ = 0.5773502693;
  hexwidth_ = 6.185;
  hexside_  = 2.0 * hexwidth_ * tan30deg_;
}

HGCalTBCheckGunPostion::~HGCalTBCheckGunPostion() { }

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HGCalTBCheckGunPostion::filter(edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {
  edm::Handle<edm::HepMCProduct> hepmc;
  iEvent.getByToken(hepMCproductLabel_, hepmc);
#ifdef DebugLog
  if (verbosity_) 
    std::cout << "isHandle valid: " << isHandle valid << std::endl;
#endif
  double x(0), y(0);
   
  if (hepmc.isValid()) {
    const HepMC::GenEvent* Evt = hepmc->GetEvent();
     
#ifdef DebugLog
    if (verbosity_) 
      std::cout << "vertex " << Evt->vertices_size() << std::endl;
#endif
    for (HepMC::GenEvent::vertex_const_iterator p = Evt->vertices_begin();
	 p != Evt->vertices_end(); ++p) {
      x = (*p)->position().x()/10.; // in cm
      y = (*p)->position().y()/10.; // in cm
#ifdef DebugLog
      z = (*p)->position().z()/10.; // in cm
      if (verbosity_) 
	std::cout << " x: " << (*p)->position().x() << ":" << x
		  << " y: " << (*p)->position().y() << ":" << y
		  << " z: " << (*p)->position().z() << ":" << z
		  << std::endl;
#endif
    }
  }//if (genEventInfoHandle.isValid())

  bool   flag(false);
  if (method2_) {
    bool cond1 = y==0 && x >=(-hexside_ * sqrt(3)/2.);
    bool cond2 = ((y+hexside_) >= -x/sqrt(3)) && (y<0&&x<0);
    bool cond3 = (y*sqrt(3) >= (x - hexside_*sqrt(3))) && (y<0 && x>0);
    bool cond4 = y==0 && x <=(hexside_ * sqrt(3)/2.);
    bool cond5 = (-y*sqrt(3) >= (x - hexside_*sqrt(3))) && (y>0&&x>0);
    bool cond6 = ((y-hexside_) <= x/sqrt(3)) && (y>0&&x<0);
    flag = cond1 || cond2 || cond3 || cond4 || cond5 || cond6;
  } else {
    double absx = std::abs(x);
    double absy = std::abs(y);
    if (absx <= hexwidth_ && absy <= hexside_) {
      if (absy <= hexwidth_*tan30deg_ || 
	  absx <= (2.*hexwidth_ - absy/tan30deg_)) flag = true;
    }
  }
  
#ifdef DebugLog
  if (verbosity_) std::cout << "Selection Flag " << flag << std::endl;
#endif
  return flag;
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalTBCheckGunPostion::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTBCheckGunPostion);
