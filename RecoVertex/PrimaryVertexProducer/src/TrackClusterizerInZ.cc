#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"

using namespace std;

namespace {

  bool recTrackLessZ(const reco::TransientTrack & tk1, 
		     const reco::TransientTrack & tk2) 
  {
    return tk1.initialFreeState().position().z() < tk2.initialFreeState().position().z();
  }


  bool beamTrackLessZ(const BeamTransientTrack & tk1, 
		      const BeamTransientTrack & tk2) 
  {
    return tk1.zBeam() < tk2.zBeam();
  }
}


TrackClusterizerInZ::TrackClusterizerInZ(const edm::ParameterSet& conf) 
  : theConfig(conf) {}


vector< vector<reco::TransientTrack> > 
TrackClusterizerInZ::clusterize(const vector<reco::TransientTrack> & tracks) 
  const 
{

  vector<reco::TransientTrack> tks = tracks; // copy to be sorted
  
  vector< vector<reco::TransientTrack> > clusters;
  if (tks.empty()) return clusters;

  // sort in increasing order of z
  stable_sort(tks.begin(), tks.end(), recTrackLessZ);

  // init first cluster
  vector<reco::TransientTrack>::const_iterator it = tks.begin();
  vector <reco::TransientTrack> currentCluster; currentCluster.push_back(*it);

  it++;
  for ( ; it != tks.end(); it++) {

    double zPrev = currentCluster.back().initialFreeState().position().z();
    double zCurr = (*it).initialFreeState().position().z();
    if ( abs(zCurr - zPrev) < zSeparation() ) {
      // close enough ? cluster together
      currentCluster.push_back(*it);
    }
    else {
      // store current cluster, start new one
      clusters.push_back(currentCluster);
      currentCluster.clear();
      currentCluster.push_back(*it);
      it++; if (it == tks.end()) break;
    }
  }

  // store last cluster
  clusters.push_back(currentCluster);

  return clusters;

}



vector< vector<reco::TransientTrack> > 
TrackClusterizerInZ::clusterize(const vector<BeamTransientTrack> & tracks) 
  const 
{

  vector<BeamTransientTrack> tks = tracks; // copy to be sorted
  
  vector< vector<reco::TransientTrack> > clusters;
  if (tks.empty()) return clusters;

  // sort in increasing order of z
  stable_sort(tks.begin(), tks.end(), beamTrackLessZ);

  // init first cluster
  vector<BeamTransientTrack>::const_iterator it = tks.begin();
  vector <reco::TransientTrack> currentCluster; 
  
  currentCluster.push_back(*it);
  double zPrev=(*it).beamState().position().z();

  it++;
  for ( ; it != tks.end(); it++) {

    double zCurr = (*it).beamState().position().z();
    if ( abs(zCurr - zPrev) < zSeparation() ) {
      // close enough ? cluster together
      currentCluster.push_back(*it);
      zPrev=zCurr;
    }
    else {
      // store current cluster, start new one
      clusters.push_back(currentCluster);
      currentCluster.clear();
      currentCluster.push_back(*it);
      zPrev = (*it).beamState().position().z();

      it++; if (it == tks.end()) break;
    }
  }

  // store last cluster
  clusters.push_back(currentCluster);

  return clusters;

}


float TrackClusterizerInZ::zSeparation() const 
{
  return theConfig.getParameter<double>("zSeparation");
}
