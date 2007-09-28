#ifndef TrajectoryToSeeds_classes_h
#define TrajectoryToSeeds_classes_h

#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrackCandidateToTrajectoryMap.h"
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


         TrackCandidateToTrajectoryMap  amp3;
          ///edm::helpers::KeyVal<edm::RefProd<TrajectoryCollection>,edm::RefProd<TrajectorySeedCollection > > kv1;
          edm::Wrapper<TrackCandidateToTrajectoryMap> ampw3;
          TrackCandidateToTrajectoryMap  tts3;
          TrackCandidateToTrajectoryMapRef  ttsmref3;
          TrackCandidateToTrajectoryMapRefProd ttsmrefprod3;
          TrackCandidateToTrajectoryMapRefVector ttsmrefvec3;


         TrajectoryToTrajectoryMap  amp4;
          edm::Wrapper<TrajectoryToTrajectoryMap> ampw4;
          TrajectoryToTrajectoryMap  tts4;
          TrajectoryToTrajectoryMapRef  ttsmref4;
          TrajectoryToTrajectoryMapRefProd ttsmrefprod4;
          TrajectoryToTrajectoryMapRefVector ttsmrefvec4;


         TrackToTracksMap  amp5;
          edm::Wrapper<TrackToTracksMap> ampw5;
          TrackToTracksMap  tts5;
          TrackToTracksMapRef  ttsmref5;
          TrackToTracksMapRefProd ttsmrefprod5;
          TrackToTracksMapRefVector ttsmrefvec5;

         TrackToTrajectoryMap  amp6;
          edm::Wrapper<TrackToTrajectoryMap> ampw6;
          TrackToTrajectoryMap  tts6;
          TrackToTrajectoryMapRef  ttsmref6;
          TrackToTrajectoryMapRefProd ttsmrefprod6;
          TrackToTrajectoryMapRefVector ttsmrefvec6;

	  
	

        }
}

#endif

