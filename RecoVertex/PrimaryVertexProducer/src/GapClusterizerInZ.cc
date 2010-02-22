#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/PrimaryVertexProducer/interface/GapClusterizerInZ.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


using namespace std;


namespace {

  bool recTrackLessZ(const reco::TransientTrack & tk1,
                     const reco::TransientTrack & tk2)
  {
    return tk1.stateAtBeamLine().trackStateAtPCA().position().z() < tk2.stateAtBeamLine().trackStateAtPCA().position().z();
  }

}

 

GapClusterizerInZ::GapClusterizerInZ(const edm::ParameterSet& conf) 
{
  // some defaults to avoid uninitialized variables
  verbose_= conf.getUntrackedParameter<bool>("verbose", false);
  zSep = conf.getParameter<double>("zSeparation");
  if(verbose_) {std::cout << "TrackClusterizerInZ:  algorithm=gap, zSeparation="<< zSep << std::endl;}
}



float GapClusterizerInZ::zSeparation() const 
{
  return zSep;
}




vector< vector<reco::TransientTrack> >
GapClusterizerInZ::clusterize(const vector<reco::TransientTrack> & tracks)
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

    double zPrev = currentCluster.back().stateAtBeamLine().trackStateAtPCA().position().z();
    double zCurr = (*it).stateAtBeamLine().trackStateAtPCA().position().z();

    if ( abs(zCurr - zPrev) < zSeparation() ) {
      // close enough ? cluster together
      currentCluster.push_back(*it);
    }
    else {
      // store current cluster, start new one
      clusters.push_back(currentCluster);
      currentCluster.clear();
      currentCluster.push_back(*it);
    }
  }

  // store last cluster
  clusters.push_back(currentCluster);

  return clusters;

}


