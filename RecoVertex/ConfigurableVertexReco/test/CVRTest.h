#ifndef RecoBTag_CVRTest
#define RecoBTag_CVRTest

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTS.h"

class TNtuple;

class CVRTest : public edm::EDAnalyzer {
  /**
   *  Class that glues the combined btagging algorithm to the framework
   */
   public:
      explicit CVRTest( const edm::ParameterSet & );
      ~CVRTest();

      virtual void analyze( const edm::Event &, const edm::EventSetup &);

      reco::TransientTrack TrackMove(const reco::TransientTrack &, const float &, const float &, const 
	float &);

   private:
      void discussPrimary( const edm::Event & ) const;

   private:
      ConfigurableVertexReconstructor * vrec_;
      std::string trackcoll_;
      std::string vertexcoll_;
      std::string beamspot_;
      TNtuple * tree_;	
      GlobalTrajectoryParameters * move_param;
      GlobalPoint * move_pos;
      FreeTrajectoryState * move_traj;
      reco::TransientTrack * move_track;

};

#endif
