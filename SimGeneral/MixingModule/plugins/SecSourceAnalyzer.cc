// -*- C++ -*-
//
// Package:    SecSourceAnalyzer
// Class:      SecSourceAnalyzer
// 
/**\class SecSourceAnalyzer SecSourceAnalyzer.cc SecSource/SecSourceAnalyzer/src/SecSourceAnalyzer.cc

 Description: Get the data from the secondary source file using the getProductByTag method

 Implementation:
   
*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Wed Apr 22 16:54:31 CEST 2009
// $Id: SecSourceAnalyzer.cc,v 1.8 2012/01/30 18:23:36 fwyzard Exp $
//
//


// system include files
#include <memory>
#include <boost/bind.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"

#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"

#include "FWCore/Sources/interface/VectorInputSource.h"

#include "Mixing/Base/interface/PileUp.h"

#include "SecSourceAnalyzer.h"

#include "TH1F.h"

//
// constructors and destructor
//
namespace edm
{
SecSourceAnalyzer::SecSourceAnalyzer(const edm::ParameterSet& iConfig)
  :minBunch_(0),
   maxBunch_(0),
   tag_(InputTag())
{
//    int minb = minBunch_;
//    int maxb = maxBunch_;
   int averageNumber = 1;
   std::string histoFileName = " ";
   TH1F * histoName = new TH1F("h","",10,0,10); 
   bool playback = false;
   
   input_.reset(new edm::PileUp(iConfig.getParameter<edm::ParameterSet>("input"),
                                averageNumber,histoName,playback));
      
   dataStep2_ = iConfig.getParameter<bool>("dataStep2");
   
   
   if (dataStep2_)
     // The data file contain the PCrossingFrame<SimTrack>
     label_   = iConfig.getParameter<edm::InputTag>("collPCF");
   else
     // The data file contain the SimTrack
     label_   = iConfig.getParameter<edm::InputTag>("collSimTrack");

}


SecSourceAnalyzer::~SecSourceAnalyzer()
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
SecSourceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   vectorEventIDs_.resize(maxBunch_-minBunch_+1);

   int nevt = 0 ;
   for( int ibx = minBunch_ ; ibx <= maxBunch_ ; ++ibx )
     {
       if( ibx == 0 )
	 {
	   input_->readPileUp( iEvent.id(),
                               vectorEventIDs_[ ibx-minBunch_ ],
			       boost::bind(&SecSourceAnalyzer::getBranches, 
					   this, _1), ibx
			       );
	 }
       else
	 {
	   input_->readPileUp( iEvent.id(),
                               vectorEventIDs_[ ibx-minBunch_ ],
			       boost::bind(&SecSourceAnalyzer::dummyFunction, 
					   this, _1), ibx
			       );
	 }

       nevt += vectorEventIDs_[ ibx-minBunch_ ].size() ;
     }

   std::cout << "-> The std::vector<EventPrincipalVector> of the secondary source 'input' has been filled with " 
   	     << nevt << " element corresponding to " << maxBunch_-minBunch_+1 
	     << " bunch." << std::endl;
}

void  SecSourceAnalyzer::getBranches(EventPrincipal const &ep)
  { 
    std::cout <<"-> Get the event:  id " << ep.id() << std::endl;
    std::cout << "-> dataStep2_ = " << dataStep2_ << std::endl;
    tag_ = InputTag(label_);
    
    std::cout << "-> Will try to get the branch with the tag : " << tag_ << std::endl;    
    std::cout << " and the EventPrincipal ep with a size = " << ep.size() << std::endl;
    
    if (!dataStep2_){
        // Get the SimTrack collection
        
	// default version changed to transmit vertexoffset
        boost::shared_ptr<Wrapper<std::vector<SimTrack> > const> shPtr =
        getProductByTag<std::vector<SimTrack> >(ep, tag_);
    
        if (shPtr) 
		std::cout << "-> Could get SimTrack !" << std::endl;
	else 
		std::cout << "-> Could not get SimTrack !" << std::endl;

    }
    else{
        // Get the PCrossingFrame collection given as signal
    
        // default version changed to transmit vertexoffset
	tag_ = InputTag("CFwriter","g4SimHits");
        boost::shared_ptr<Wrapper<PCrossingFrame<SimTrack> > const> shPtr =
        getProductByTag<PCrossingFrame<SimTrack> >(ep, tag_);
        
	if (shPtr) 
		std::cout << "-> Could get PCrossingFrame<SimTrack> !" << std::endl;
	else 
		std::cout << "-> Could not get PCrossingFrame<SimTrack> !" << std::endl;
	
    }
 
  }


// ------------ method called once each job just before starting event loop  ------------
void 
SecSourceAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SecSourceAnalyzer::endJob() {
  if (input_) input_->endJob();
}

}//edm
