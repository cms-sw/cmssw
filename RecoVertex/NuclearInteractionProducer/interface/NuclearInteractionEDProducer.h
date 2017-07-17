#ifndef CD_NuclearInteractionEDProducer_H_
#define CD_NuclearInteractionEDProducer_H_
// -*- C++ -*-
//
// Package:    NuclearAssociatonMapEDProducer
// Class:      NuclearInteractionEDProducer
//
/**\class NuclearInteractionEDProducer NuclearInteractionEDProducer.h RecoVertex/NuclearInteractionProducer/interface/NuclearInteractionEDProducer.h

 Description: Associate nuclear seeds to primary tracks and associate secondary tracks to primary tracks

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincent ROBERFROID
//         Created:  Fri Aug 10 12:05:36 CET 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/VertexReco/interface/NuclearInteraction.h"

class NuclearVertexBuilder;
class NuclearLikelihood;

class NuclearInteractionEDProducer : public edm::stream::EDProducer<> {

public:
      typedef edm::RefVector<TrajectorySeedCollection> TrajectorySeedRefVector;

      explicit NuclearInteractionEDProducer(const edm::ParameterSet&);
      ~NuclearInteractionEDProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

      static bool isInside( const reco::TrackRef& track, const TrajectorySeedRefVector& seeds);
      void findAdditionalSecondaryTracks( reco::NuclearInteraction& nucl,
                                          const edm::Handle<reco::TrackCollection>& additionalSecTracks) const;

      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      edm::EDGetTokenT<reco::TrackCollection>	 token_primaryTrack; 
      edm::EDGetTokenT<reco::TrackCollection>	 token_secondaryTrack; 
      edm::EDGetTokenT<reco::TrackCollection>	 token_additionalSecTracks; 
      edm::EDGetTokenT<TrajectoryCollection> token_primaryTrajectory; 
      edm::EDGetTokenT<TrajTrackAssociationCollection> token_refMapH; 
      edm::EDGetTokenT<TrajectoryToSeedsMap> token_nuclMapH; 

      std::unique_ptr< NuclearVertexBuilder >  vertexBuilder;
      std::unique_ptr< NuclearLikelihood >     likelihoodCalculator;

  edm::ESWatcher<IdealMagneticFieldRecord> magFieldWatcher_;
  edm::ESWatcher<TransientTrackRecord> transientTrackWatcher_;

};

#endif
