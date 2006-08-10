#include "TrackingTools/TrackAssociator/interface/TestMuon.h"
using namespace reco;

TestMuon::TestMuon( double chi2, double ndof,
		    const ParameterVector & par, double pt, const CovarianceMatrix & cov ) :
  TrackBase( chi2, ndof, par, pt, cov ) 
{
}
