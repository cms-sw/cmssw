#ifndef TPFakeRateSelector_h
#define TPFakeRateSelector_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
class Event;
}


class TPFakeRateSelector {
  typedef TrackingParticleCollection collection;
  typedef std::vector<const TrackingParticle *> container;
  typedef container::const_iterator const_iterator;

public:
  TPFakeRateSelector ( const edm::ParameterSet & cfg ) :
    ptMin_( cfg.getParameter<double>( "ptMin" ) ),
    minRapidity_( cfg.getParameter<double>( "minRapidity" ) ),
    maxRapidity_( cfg.getParameter<double>( "maxRapidity" ) ),
    tip_( cfg.getParameter<double>( "tip" ) ),
    lip_( cfg.getParameter<double>( "lip" ) ),
    minHit_( cfg.getParameter<int>( "minHit" ) ) 
  { }
  
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }

  void select(const collection & c, const edm::Event & ) {
    selected_.clear();
    for( collection::const_iterator tp = c.begin(); 
         tp != c.end(); ++ tp )
      if ( makeSelection(* tp) ) selected_.push_back( & * tp );
  }
  bool operator()( const TrackingParticle & tp ) const { 
    return  makeSelection( tp );
  }

private:
  bool makeSelection( const TrackingParticle & tp ) const { 
    return  
      (tp.matchedHit() >= minHit_ &&
       sqrt(tp.momentum().perp2()) >= ptMin_ && 
       fabs(tp.momentum().eta()) >= minRapidity_ && fabs(tp.momentum().eta()) <= maxRapidity_ && 
       fabs(tp.parentVertex()->position().perp()) <= tip_ &&
       fabs(tp.parentVertex()->position().z()) <= lip_ );
  }
  
  container selected_;
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;

};

#endif
