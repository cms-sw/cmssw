// -*- C++ -*-
//
// Package:    ShashlikSimHitValidation
// Class:      ShashlikSimHitValidation
// 
/**\class ShashlikSimHitValidation ShashlikSimHitValidation.cc Validation/ShashlikValidation/plugins/ShashlikSimHitValidation.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Shilpi Jain
//         Created:  Tue, 04 Mar 2014 14:17:25 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

//
// class declaration
//

class ShashlikSimHitValidation : public edm::EDAnalyzer {

public:
  explicit ShashlikSimHitValidation(const edm::ParameterSet&);
  ~ShashlikSimHitValidation();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  /*
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  */

  // ----------member data ---------------------------
  std::string           caloHitSource_;
  DQMStore              *dbe_;
  int                   verbosity_;
  std::map<std::string,MonitorElement*> histmap;
  std::map<std::string,MonitorElement*> histmap2d;
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
ShashlikSimHitValidation::ShashlikSimHitValidation(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  dbe_           = edm::Service<DQMStore>().operator->();
  caloHitSource_ = iConfig.getParameter<std::string>("CaloHitSource");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);
  if (verbosity_ > 0) std::cout << "Start SimHitValidation for PCaloHit at "
				<< caloHitSource_ << std::endl;
}  

ShashlikSimHitValidation::~ShashlikSimHitValidation() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------
void ShashlikSimHitValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<edm::PCaloHitContainer> pcalohit;
  iEvent.getByLabel("g4SimHits", caloHitSource_, pcalohit);
  const edm::PCaloHitContainer* endcapHits = pcalohit.product();
  
  double sumhitE = 0;
  unsigned int kount(0);
  for (edm::PCaloHitContainer::const_iterator erh = endcapHits->begin(); 
       erh != endcapHits->end(); erh++) {
    
    double time          = erh->time();
    double energy        = erh->energy();
    
    uint32_t detidindex  = erh->id();
    EKDetId detid(detidindex);
    int ix = detid.ix();
    int iy = detid.iy();
    int iz = detid.zside();

    sumhitE += energy;
    if (dbe_) {
      histmap["time"]->Fill(time);
      histmap["timeEwei"]->Fill(time,energy);
      histmap["energy"]->Fill(energy);
    
      histmap2d["iyVSix"]->Fill(ix,iy);
      histmap2d["iyVSixEwei"]->Fill(ix,iy,energy);
    
      if (time>0 && time<100) {
	histmap2d["iyVSixEwei_100"]->Fill(ix,iy,energy);
	if (time>0 && time<25) {
	  histmap2d["iyVSixEwei_25"]->Fill(ix,iy,energy);
	}
      }

      if (iz==-1) {       ///same for zside = -1
	histmap2d["iyVSix_zM"]->Fill(ix,iy);
	histmap2d["iyVSixEwei_zM"]->Fill(ix,iy,energy);
	
	if (time>0 && time<100) {
	  histmap2d["iyVSixEwei_100_zM"]->Fill(ix,iy,energy);
	  if (time>0 && time<25) {
	    histmap2d["iyVSixEwei_25_zM"]->Fill(ix,iy,energy);
	  }
	}
      } else if (iz==1) { ///same for zside = +1
	histmap2d["iyVSix_zP"]->Fill(ix,iy);
	histmap2d["iyVSixEwei_zP"]->Fill(ix,iy,energy);
	
	if (time>0 && time<100) {
	  histmap2d["iyVSixEwei_100_zP"]->Fill(ix,iy,energy);
	  if (time>0 && time<25) {
	    histmap2d["iyVSixEwei_25_zP"]->Fill(ix,iy,energy);
	  }
	}
      }
    }
    kount++;
  }
  if (dbe_) histmap["sumE"]->Fill(sumhitE);
  if (verbosity_ > 1) std::cout << kount << " hits for " << caloHitSource_
				<< " with total energy " << sumhitE << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void ShashlikSimHitValidation::beginJob() {

  if (dbe_) {
    histmap["time"] = dbe_->book1D("time","Time distribution of the hits",100,0.,530);
    histmap["timeEwei"] = dbe_->book1D("timeEwei","Energy weighted time distribution of the hits",100,0,530);
    histmap["energy"] = dbe_->book1D("energy","Energy distribution of the hits",100,0,3);
    histmap["sumE"] = dbe_->book1D("sumE","Total energy distribution of the hits",500,0,100);

    histmap2d["iyVSix"] = dbe_->book2D("iyVSix","iy VS ix",301,0,301,301,0,301);
    histmap2d["iyVSixEwei"] = dbe_->book2D("iyVSixEwei","iy VS ix (Energyweighted)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_25"] = dbe_->book2D("iyVSixEwei_25","iy VS ix (Energyweighted) in a time window of 0 to 25 ns",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_100"] = dbe_->book2D("iyVSixEwei_100","iy VS ix (Energyweighted) in a time window of 0 to 100 ns",301,0,301,301,0,301);

    //iz=-1
    histmap2d["iyVSix_zM"] = dbe_->book2D("iyVSix_zM","iy VS ix (-tive z)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_zM"] = dbe_->book2D("iyVSixEwei_zM","iy VS ix (Energyweighted)(-tive z)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_25_zM"] = dbe_->book2D("iyVSixEwei_25_zM","iy VS ix (Energyweighted) in a time window of 0 to 25 ns (-tive z)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_100_zM"] = dbe_->book2D("iyVSixEwei_100_zM","iy VS ix (Energyweighted) in a time window of 0 to 100 ns (-tive z)",301,0,301,301,0,301);
    
    //iz=+1
    histmap2d["iyVSix_zP"] = dbe_->book2D("iyVSix_zP","iy VS ix (+tive z)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_zP"] = dbe_->book2D("iyVSixEwei_zP","iy VS ix (Energyweighted)(+tive z)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_25_zP"] = dbe_->book2D("iyVSixEwei_25_zP","iy VS ix (Energyweighted) in a time window of 0 to 25 ns (+tive z)",301,0,301,301,0,301);
    histmap2d["iyVSixEwei_100_zP"] = dbe_->book2D("iyVSixEwei_100_zP","iy VS ix (Energyweighted) in a time window of 0 to 100 ns (+tive z)",301,0,301,301,0,301);

    if (verbosity_>0) 
      std::cout << "ShashlikSimHitValidation:: Initialize histograms" << std::endl;
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void ShashlikSimHitValidation::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void ShashlikSimHitValidation::beginRun(edm::Run const&, edm::EventSetup const&) {
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void ShashlikSimHitValidation::endRun(edm::Run const&, edm::EventSetup const&) {
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void ShashlikSimHitValidation::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void ShashlikSimHitValidation::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ShashlikSimHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ShashlikSimHitValidation);
