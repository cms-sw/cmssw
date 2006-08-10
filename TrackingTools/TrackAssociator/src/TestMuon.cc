#include "TrackingTools/TrackAssociator/interface/TestMuon.h"

using namespace std;
using namespace reco;

TestMuon::TestMuon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	    const Parameters & p, const Covariance & c ) :
  TrackBase( chi2, ndof, found, invalid, lost, p, c ) {
}

