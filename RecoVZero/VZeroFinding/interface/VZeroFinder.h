#ifndef _VZeroFinder_h_
#define _VZeroFinder_h_

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VZero/interface/VZero.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

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

   GlobalVector rotate(const GlobalVector & p, double a);
   bool checkTrackPair(const reco::Track& posTrack,
                       const reco::Track& negTrack,
                       const reco::VertexCollection* vertices,
                       reco::VZeroData& data);

 private:
   GlobalTrajectoryParameters getGlobalTrajectoryParameters(const reco::Track& track);

   float maxDca,
         minCrossingRadius,
         maxCrossingRadius,
         maxImpactMother; 

   const MagneticField * theMagField;
};

#endif

