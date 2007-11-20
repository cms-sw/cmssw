#ifndef CD_NuclearInteractionEDProducer_H_
#define CD_NuclearInteractionEDProducer_H_
// -*- C++ -*-
//
// Package:    NuclearAssociatonMapEDProducer
// Class:      NuclearInteractionEDProducer
//
/**\class NuclearInteractionEDProducer NuclearInteractionEDProducer.h RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionEDProducer.h

 Description: Associate nuclear seeds to primary tracks and associate secondary tracks to primary tracks

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincent ROBERFROID
//         Created:  Fri Aug 10 12:05:36 CET 2007
// $Id: NuclearInteractionEDProducer.h,v 1.2 2007/09/05 15:15:28 roberfro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"


class NuclearInteractionEDProducer : public edm::EDProducer {

public:
      typedef edm::RefVector<TrajectorySeedCollection> TrajectorySeedRefVector;

      explicit NuclearInteractionEDProducer(const edm::ParameterSet&);
      ~NuclearInteractionEDProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

      bool isInside( const reco::TrackRef& track, const TrajectorySeedRefVector& seeds);

      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      std::string primaryProducer_;
      std::string seedsProducer_;
      std::string secondaryProducer_;
      const MagneticField * theMagField;

};
#endif
