// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      MuonIdProducer
// 
/*

 Description: Create a new collection of muons filling muon ID information.
              reco::TrackCollection or reco::MuonCollection can be used as input.

 Implementation:

*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonIdProducer.cc,v 1.3 2006/08/14 16:19:37 dmytro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "TrackingTools/TrackAssociator/interface/TrackAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TestMuon.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

#include <boost/regex.hpp>
#include "TrackingTools/TrackAssociator/interface/MuonIdProducer.h"

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig)
{
   outputCollectionName_ = iConfig.getParameter<std::string>("outputCollection");
   produces<reco::TestMuonCollection>(outputCollectionName_);

   useEcal_ = iConfig.getParameter<bool>("useEcal");
   useHcal_ = iConfig.getParameter<bool>("useHcal");
   useMuon_ = iConfig.getParameter<bool>("useMuon");
   minPt_ = iConfig.getParameter<double>("minPt");
   maxRfromIP_ = iConfig.getParameter<double>("maxDistanceFromIP");
   
   // Fill data labels
   std::vector<std::string> labels = iConfig.getParameter<std::vector<std::string> >("labels");
   boost::regex regExp1 ("([^\\s,]+)[\\s,]+([^\\s,]+)$");
   boost::regex regExp2 ("([^\\s,]+)[\\s,]+([^\\s,]+)[\\s,]+([^\\s,]+)$");
   boost::smatch matches;
	
   for(std::vector<std::string>::const_iterator label = labels.begin(); label != labels.end(); label++) {
      if (boost::regex_match(*label,matches,regExp1))
	trackAssociator_.addDataLabels(matches[1],matches[2]);
      else if (boost::regex_match(*label,matches,regExp2))
	trackAssociator_.addDataLabels(matches[1],matches[2],matches[3]);
      else
	edm::LogError("ConfigurationError") << "Failed to parse label:\n" << *label << "Skipped.\n";
   }
   
   // Determine input collection
   std::string inputCollection = iConfig.getParameter<std::string>("inputCollection");
   if (boost::regex_match(inputCollection,matches,regExp1))
     {
	inputCollectionType_ = matches[1];
	inputCollectionLabels_ = std::pair<std::string,std::string>(matches[2],"");
     }
   else if (boost::regex_match(inputCollection,matches,regExp2))
     {
	inputCollectionType_ = matches[1];
	inputCollectionLabels_ = std::pair<std::string,std::string>(matches[2],matches[3]);
     }
   else
     throw cms::Exception("FatalError") << "Failed to parse inputCollection:\n" << inputCollection << "\n";
   
   if (inputCollectionType_ == "TrackCollection")
     mode_ = TrackCollection;
   else if (inputCollectionType_ == "MuonCollection")
     mode_ = MuonCollection;
   else 
     throw cms::Exception("FatalError") << "Unkown input type: " << inputCollectionType_ << "\n";
   trackAssociator_.useDefaultPropagator();
}


MuonIdProducer::~MuonIdProducer()
{
  TimingReport::current()->dump(std::cout);
}

void MuonIdProducer::init(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   switch (mode_) {
    case TrackCollection:
      iEvent.getByLabel(inputCollectionLabels_.first, inputCollectionLabels_.second, trackCollectionHandle_);
      if (! trackCollectionHandle_.isValid()) throw cms::Exception("FatalError") << 
	"Cannot find input list in Event: " << inputCollectionType_ << " " << 
	inputCollectionLabels_.first << " " << inputCollectionLabels_.second << "\n";
      trackCollectionIter_ = trackCollectionHandle_->begin();
      index_ = 0;
      break;
    case MuonCollection:
      iEvent.getByLabel(inputCollectionLabels_.first, inputCollectionLabels_.second, muonCollectionHandle_);
      if (! muonCollectionHandle_.isValid()) throw cms::Exception("FatalError") << 
	"Cannot find input list in Event: " << inputCollectionType_ << " " << 
	inputCollectionLabels_.first << " " << inputCollectionLabels_.second << "\n";
      muonCollectionIter_ = muonCollectionHandle_->begin();
      break;
   }
}

reco::TestMuon* MuonIdProducer::getNewMuon(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   switch (mode_) {
    case TrackCollection:
      if( trackCollectionIter_ !=  trackCollectionHandle_->end())
	{
	   reco::TestMuon* aMuon = new reco::TestMuon;
	   aMuon->setTrack(reco::TrackRef(trackCollectionHandle_,index_));
	   index_++;
	   trackCollectionIter_++;
	   return aMuon;
	}
      else return 0;
      break;
    case MuonCollection:
      if( muonCollectionIter_ !=  muonCollectionHandle_->end())
	{
	   muonCollectionIter_++;
	   // return new reco::TestMuon(*muonCollectionIter_);
	}
      else return 0;
      break;
   }
   return 0;
}

void MuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::auto_ptr<reco::TestMuonCollection> outputMuons(new reco::TestMuonCollection);

   TimerStack timers;
   timers.push("MuonIdProducer::produce::init");
   init(iEvent, iSetup);
   timers.clean_stack();
   


   // loop over input collection
   while(reco::TestMuon* aMuon = getNewMuon(iEvent, iSetup))
     {
	LogTrace("MuonIdProducer::produce") << "-----------------" << "\n";
	LogTrace("MuonIdProducer::produce") << "(Pt: " << aMuon->track().get()->pt() << " GeV" <<"\n";
	LogTrace("MuonIdProducer::produce") << "Distance from IP: " << 
	  aMuon->track().get()->vertex().rho() << " cm" <<"\n";

	if (aMuon->track().get()->pt() < minPt_)
	  { LogTrace("MuonIdProducer::produce") << "Skipped low Pt track (Pt: " << aMuon->track().get()->pt() << " GeV)\n";}
	else if (aMuon->track().get()->vertex().rho() > maxRfromIP_)
	  {LogTrace("MuonIdProducer::produce") << "Skipped track originated away from IP: " << 
	       aMuon->track().get()->vertex().rho() << " cm\n";
	  }
	else {
	   fillMuonId(iEvent, iSetup, *aMuon);
	   outputMuons->push_back(*aMuon);
	}
	delete aMuon;
     }
   iEvent.put(outputMuons,outputCollectionName_);
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent, const edm::EventSetup& iSetup,
				reco::TestMuon& aMuon)
{
   TrackAssociator::AssociatorParameters parameters;
   parameters.useEcal = useEcal_ ;
   parameters.useHcal = useHcal_ ;
   parameters.useMuon = useMuon_ ;
   parameters.dRHcal = 0.4;
   parameters.dRHcal = 0.4;

   TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, 
						       trackAssociator_.getFreeTrajectoryState(iSetup, *(aMuon.track().get()) ),
						       parameters);
   reco::TestMuon::MuonEnergy muonEnergy;
   muonEnergy.had = info.hcalEnergy();
   muonEnergy.em = info.ecalEnergy();
   muonEnergy.ho = info.outerHcalEnergy();
   aMuon.setCalEnergy( muonEnergy );
      
   reco::TestMuon::MuonIsolation muonIsolation;
   muonIsolation.hCalEt01 = 0;
   muonIsolation.eCalEt01 = 0;
   muonIsolation.hCalEt04 = info.hcalConeEnergy();
   muonIsolation.eCalEt04 = info.ecalConeEnergy();
   muonIsolation.hCalEt07 = 0;
   muonIsolation.eCalEt07 = 0;
   muonIsolation.trackSumPt01 = 0;
   muonIsolation.trackSumPt04 = 0;
   muonIsolation.trackSumPt07 = 0;
   aMuon.setIsolation( muonIsolation );
      
   std::vector<reco::TestMuon::MuonMatch> muonMatches;
   for( std::vector<MuonSegmentMatch>::const_iterator segment=info.segments.begin();
	segment!=info.segments.end(); segment++ )
     {
	reco::TestMuon::MuonMatch aMatch;
	aMatch.dX = segment->segmentLocalPosition.x()-segment->trajectoryLocalPosition.x();
	aMatch.dY = segment->segmentLocalPosition.y()-segment->trajectoryLocalPosition.y();
	aMatch.dXErr = sqrt(segment->trajectoryLocalErrorXX+segment->segmentLocalErrorXX);
	aMatch.dYErr = sqrt(segment->trajectoryLocalErrorYY+segment->segmentLocalErrorYY);
	aMatch.dXdZ = 0;
	aMatch.dYdZ = 0;
	aMatch.dXdZErr = 0;
	aMatch.dYdZErr = 0;
	muonMatches.push_back(aMatch);
	LogTrace("MuonIdProducer::fillMuonId")<< "Muon match (dX,dY,dXErr,dYErr): " << aMatch.dX << " \t" << aMatch.dY 
	  << " \t" << aMatch.dXErr << " \t" << aMatch.dYErr << "\n";
     }
   aMuon.setMatches(muonMatches);
   LogTrace("MuonIdProducer::fillMuonId") << "number of muon matches: " << aMuon.matches().size() << "\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdProducer)
