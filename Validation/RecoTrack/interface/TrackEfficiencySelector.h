#ifndef TrackEfficiencySelector_h
#define TrackEfficiencySelector_h

/** \class TrackEfficiencySelector
 *  Use RecoTrackSelector instead
 *
 *  $Date: 2007/03/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
class Event;
}

class TrackEfficiencySelector {

public:
  TrackEfficiencySelector ( const edm::ParameterSet & cfg ) :
    ptMin_( cfg.getParameter<double>( "ptMin" ) ),
    minRapidity_( cfg.getParameter<double>( "minRapidity" ) ),
    maxRapidity_( cfg.getParameter<double>( "maxRapidity" ) ),
    tip_( cfg.getParameter<double>( "tip" ) ),
    lip_( cfg.getParameter<double>( "lip" ) ),
    minHit_( cfg.getParameter<int>( "minHit" ) ) 
  { }
  
  bool operator()( const reco::Track & t ) {
    return
      (t.numberOfValidHits() >= minHit_ &&
       fabs(t.pt()) >= ptMin_ &&
       t.eta() >= minRapidity_ && t.eta() <= maxRapidity_ &&
       fabs(t.d0()) <= tip_ &&
       fabs(t.dz()) <= lip_ );
  }

private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;

};

#endif
