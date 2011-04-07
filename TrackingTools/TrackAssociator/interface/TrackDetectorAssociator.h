#ifndef TrackAssociator_TrackDetectorAssociator_h
#define TrackAssociator_TrackDetectorAssociator_h 1

// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TrackDetectorAssociator
// 
/*

 Description: main class of tools to associate a track to calorimeter and muon detectors

*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: TrackDetectorAssociator.h,v 1.19 2011/04/07 08:17:31 innocent Exp $
//
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

#include "DataFormats/TrackReco/interface/Track.h"


class FreeTrajectoryState;
class SimTrack;
class SimVertex;
class Propagator;

class TrackDetectorAssociator {
 public:
  TrackDetectorAssociator(){}
  virtual ~TrackDetectorAssociator(){}
   
   typedef TrackAssociatorParameters AssociatorParameters;
   enum Direction { Any, InsideOut, OutsideIn };
   
   /// propagate a track across the whole detector and
   /// find associated objects. Association is done in
   /// two modes 
   ///  1) an object is associated to a track only if 
   ///     crossed by track
   ///  2) an object is associated to a track if it is
   ///     withing an eta-phi cone of some radius with 
   ///     respect to a track.
   ///     (the cone origin is at (0,0,0))
   /// Trajectory bending in eta-phi is taking into account
   /// when matching is performed
   ///
   /// associate using FreeTrajectoryState
   virtual TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const AssociatorParameters& )=0;
   /// associate using inner and outer most states of a track
   /// in the silicon tracker. 
   virtual TrackDetMatchInfo            associate( const edm::Event& iEvent,
					   const edm::EventSetup& iSetup,
					   const AssociatorParameters& parameters,
					   const FreeTrajectoryState* innerState,
					   const FreeTrajectoryState* outerState=0)=0;
   /// associate using reco::Track
   virtual TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const reco::Track&,
					   const AssociatorParameters&,
					   Direction direction = Any )=0;
   /// associate using a simulated track
   virtual TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const SimTrack&,
					   const SimVertex&,
					   const AssociatorParameters& )=0;
   /// associate using 3-momentum, vertex and charge
   virtual TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const GlobalVector&,
					   const GlobalPoint&,
					   const int,
					   const AssociatorParameters& )=0;
   
   /// use a user configured propagator
   virtual void setPropagator( const Propagator* )=0;
   
   /// use the default propagator
   virtual void useDefaultPropagator()=0;
   

};
#endif
