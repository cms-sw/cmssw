#ifndef TrajectoryToSeeds_classes_h
#define TrajectoryToSeeds_classes_h

#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

namespace {
   namespace {
          TrajectoryToSeedsMap  amp1;
          ///edm::helpers::KeyVal<edm::RefProd<TrajectoryCollection>,edm::RefProd<TrajectorySeedCollection > > kv1;
          edm::Wrapper<TrajectoryToSeedsMap> ampw1;
          TrajectoryToSeeds  tts1;
          TrajectoryToSeedsMapRef  ttsmref1;
          TrajectoryToSeedsMapRefProd ttsmrefprod1;
          TrajectoryToSeedsMapRefVector ttsmrefvec1;

         TrackToSeedsMap  amp2;
          ///edm::helpers::KeyVal<edm::RefProd<TrajectoryCollection>,edm::RefProd<TrajectorySeedCollection > > kv1;
          edm::Wrapper<TrackToSeedsMap> ampw2;
          TrackToSeeds  tts2;
          TrackToSeedsMapRef  ttsmref2;
          TrackToSeedsMapRefProd ttsmrefprod2;
          TrackToSeedsMapRefVector ttsmrefvec2;

        }
}

#endif