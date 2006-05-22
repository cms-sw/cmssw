#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"

using namespace std;

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  return (tk.pt() > minPt() 
	  && abs(tk.d0() / tk.d0Error()) < maxD0Significance());
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
