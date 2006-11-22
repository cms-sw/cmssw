#ifndef TPEfficiencySelector_h
#define TPEfficiencySelector_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
class Event;
}


class TPEfficiencySelector {

public:
  TPEfficiencySelector ( const edm::ParameterSet & cfg ) :
    ptMin_( cfg.getParameter<double>( "ptMin" ) ),
    minRapidity_( cfg.getParameter<double>( "minRapidity" ) ),
    maxRapidity_( cfg.getParameter<double>( "maxRapidity" ) ),
    tip_( cfg.getParameter<double>( "tip" ) ),
    lip_( cfg.getParameter<double>( "lip" ) ),
    minHit_( cfg.getParameter<int>( "minHit" ) ) 
  { }
  
  bool operator()( const TrackingParticle & tp ) const { 
    return (
	    tp.matchedHit() >= minHit_ &&
	    sqrt(tp.momentum().perp2()) >= ptMin_ && 
	    fabs(tp.momentum().eta()) >= minRapidity_ && fabs(tp.momentum().eta()) <= maxRapidity_ && 
	    fabs(tp.parentVertex()->position().perp()) <= tip_ &&
	    fabs(tp.parentVertex()->position().z()) <= lip_ 
	    );
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
