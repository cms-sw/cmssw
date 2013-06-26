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
// $Id: ExtraFromSeeds.h,v 1.2 2013/02/27 13:28:54 muzaffar Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

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

class ExtraFromSeeds : public edm::EDProducer {
   public:
      explicit ExtraFromSeeds(const edm::ParameterSet&);
      ~ExtraFromSeeds();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
  edm::InputTag tracks_;
  typedef std::vector<unsigned int> ExtremeLight;

      // ----------member data ---------------------------
};

#endif
