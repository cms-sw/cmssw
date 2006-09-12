#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  std::cout << "TrackFilterForPVFinding::operator " << std::endl;
  std::cout << "pt " << tk.pt() << " " << minPt() << "   =>" << (tk.pt() > minPt()) << std::endl;
  std::cout << "d0 " << tk.d0() << " " << tk.d0Error() << " sig=" << std::abs(tk.d0() / tk.d0Error()) << " max="<< maxD0Significance() << "   =>" << (std::abs(tk.d0() / tk.d0Error()) < maxD0Significance()) << std::endl;

  PerigeeTrajectoryParameters::ParameterVector  p = tk.parameters();
  PerigeeTrajectoryError::CovarianceMatrix c = tk.covariance();
  std:: cout << "x00" <<  p[3] << " " << sqrt(c(3,3)) << " " << sqrt(tk.covariance()(3,3)) << std::endl;
  double d0Error=sqrt(c(3,3));

  return ( (tk.pt() > minPt())
	  && (std::abs(tk.d0() / d0Error) < maxD0Significance()));
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
