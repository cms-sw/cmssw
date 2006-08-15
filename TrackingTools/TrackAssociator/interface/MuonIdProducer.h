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

class MuonIdProducer : public edm::EDProducer {
 public:
   enum InputMode {TrackCollection, MuonCollection};
   explicit MuonIdProducer(const edm::ParameterSet&);
   
   ~MuonIdProducer();
   
   virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
   void               fillMuonId(edm::Event&, const edm::EventSetup&,
				 reco::TestMuon& aMuon);
   void               init(edm::Event&, const edm::EventSetup&);
   reco::TestMuon*    getNewMuon(edm::Event& iEvent, 
				 const edm::EventSetup& iSetup);

   TrackAssociator trackAssociator_;
   bool useEcal_;
   bool useHcal_;
   bool useMuon_;
   std::string inputCollectionType_;
   std::string outputCollectionName_;
   std::pair<std::string,std::string> inputCollectionLabels_;
   InputMode mode_;
   double minPt_;
   double maxRfromIP_;
   edm::Handle<reco::TrackCollection> trackCollectionHandle_;
   reco::TrackCollection::const_iterator trackCollectionIter_;
   edm::Handle<reco::MuonCollection> muonCollectionHandle_;
   reco::MuonCollection::const_iterator muonCollectionIter_;
   int index_;
};
