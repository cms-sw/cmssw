#ifndef _VZeroFinder_h_
#define _VZeroFinder_h_

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VZero/interface/VZero.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

class VZeroFinder
{
 public:
   VZeroFinder(const edm::EventSetup& es,
                    const edm::ParameterSet& pset);
   ~VZeroFinder();

   bool checkTrackPair(const reco::Track& posTrack,
                       const reco::Track& negTrack,
                       const reco::VertexCollection* vertices,
                       reco::VZeroData& data);

 private:
   FreeTrajectoryState getTrajectory(const reco::Track& track);

   ClosestApproachInRPhi * theApproach;

   float maxDcaR,
         maxDcaZ,
         minCrossingRadius,
         maxCrossingRadius,
         maxImpactMother; 

   const MagneticField * theMagField;
};

#endif

