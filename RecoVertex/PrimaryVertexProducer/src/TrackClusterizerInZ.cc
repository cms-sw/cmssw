#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"

using namespace std;

namespace {

  bool recTrackLessZ(const reco::TransientTrack & tk1, 
		     const reco::TransientTrack & tk2) 
  {
    return tk1.dz() < tk2.dz();
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

    double zPrev = currentCluster.back().dz();
    double zCurr = (*it).dz();
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


float TrackClusterizerInZ::zSeparation() const 
{
  return theConfig.getParameter<double>("zSeparation");
}
