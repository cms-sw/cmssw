#ifndef RecoTracker_TrackProducer_ExtraFromSeeds
#define RecoTracker_TrackProducer_ExtraFromSeeds

// -*- C++ -*-
//
// Package:    ExtraFromSeeds
// Class:      ExtraFromSeeds
// 
/**\class ExtraFromSeeds ExtraFromSeeds.cc RecoTracker/ExtraFromSeeds/src/ExtraFromSeeds.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Fri Feb 17 12:03:11 CET 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


//
// class declaration
//

class ExtraFromSeeds : public edm::global::EDProducer<> {
   public:
      explicit ExtraFromSeeds(const edm::ParameterSet&);
      ~ExtraFromSeeds() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<reco::TrackCollection> tracks_;
  typedef std::vector<unsigned int> ExtremeLight;

      // ----------member data ---------------------------
};

#endif
