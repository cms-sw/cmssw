#ifndef TPEfficiencySelector_h
#define TPEfficiencySelector_h

/** \class TPEfficiencySelector
 *  Filter to select TrackingParticles for efficiency studies according to pt, rapidity, tip, lip, number of hits
 *
 *  $Date: 2007/03/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
class Event;
}


class TPEfficiencySelector {

public:
  /// Constructor
  TPEfficiencySelector ( const edm::ParameterSet & cfg ) :
    ptMin_( cfg.getParameter<double>( "ptMin" ) ),
    minRapidity_( cfg.getParameter<double>( "minRapidity" ) ),
    maxRapidity_( cfg.getParameter<double>( "maxRapidity" ) ),
    tip_( cfg.getParameter<double>( "tip" ) ),
    lip_( cfg.getParameter<double>( "lip" ) ),
    minHit_( cfg.getParameter<int>( "minHit" ) ) 
  { }
  
  /// Operator() performs the selection: e.g. if (tPEfficiencySelector(tp)) {...}
  bool operator()( const TrackingParticle & tp ) const { 
    return (
	    tp.matchedHit() >= minHit_ &&
	    sqrt(tp.momentum().perp2()) >= ptMin_ && 
	    tp.momentum().eta() >= minRapidity_ && tp.momentum().eta() <= maxRapidity_ && 
	    sqrt(tp.vertex().perp2()) <= tip_ &&
 	    fabs(tp.vertex().z()) <= lip_  &&
	    //signalonly==true for efficiency
	    tp.eventId().bunchCrossing()== 0 && tp.eventId().event() == 0
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
