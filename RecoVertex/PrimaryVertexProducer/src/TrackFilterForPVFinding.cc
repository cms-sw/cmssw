#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  double d0=tk.impactPointTSCP().position().perp();
  double s0=sqrt(tk.impactPointTSCP().perigeeError().covarianceMatrix()(4,4)); 
  // note: switch to tk.impactPointTSCP().perigeeError().transverseImpactParameter when TransientTrack is fixed
  return d0<s0*maxD0Significance();
  
}

bool 
TrackFilterForPVFinding::operator() (const BeamTransientTrack & tk) const
{
  // FIXME - replace those by the beamState -- when it works
  double d0=0;//tk.initialFreeState().position().perp();
  //double s0=sqrt(tk.initialFreeState().curvilinearError().matrix()(4,4)); 
  double s0=1;

  //  std::cout << "TrackFilterForPVFinding::position " << tk.initialFreeState().position() << std::endl;
  //std::cout << "TrackFilterForPVFinding::error    " << tk.initialFreeState().curvilinearError().matrix() << std::endl;
  //std::cout << "tk.initialFreeState() " <<  tk.initialFreeState() << std::endl;
  //std::cout << "tk.beamState().FTS " <<  tk.beamState().theState() << std::endl;
  /*
  double d0=tk.beamState().position().perp();
  std::cout << "TrackFilterForPVFinding::position " << tk.initialFreeState() << std::endl;
  std::cout << "TrackFilterForPVFinding::position " << tk.beamState().position() << std::endl;
  std::cout << "TrackFilterForPVFinding::position " << tk.beamState().perigeeError().covarianceMatrix() << std::endl;
  */
  return d0<s0*maxD0Significance();
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
