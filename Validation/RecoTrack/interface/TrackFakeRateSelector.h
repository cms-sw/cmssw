#ifndef TrackFakeRateSelector_h
#define TrackFakeRateSelector_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
class Event;
}


class TrackFakeRateSelector {
  typedef reco::TrackCollection collection;
  typedef std::vector<const reco::Track *> container;
  typedef container::const_iterator const_iterator;

public:
  TrackFakeRateSelector ( const edm::ParameterSet & cfg ) :
    ptMin_( cfg.getParameter<double>( "ptMin" ) ),
    minRapidity_( cfg.getParameter<double>( "minRapidity" ) ),
    maxRapidity_( cfg.getParameter<double>( "maxRapidity" ) ),
    tip_( cfg.getParameter<double>( "tip" ) ),
    lip_( cfg.getParameter<double>( "lip" ) ),
    minHit_( cfg.getParameter<int>( "minHit" ) ) 
  { }
  
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }

  void select( 
	      const collection & c, 
	      const edm::Event & ) {
    selected_.clear();
    for( reco::TrackCollection::const_iterator trk = c.begin(); 
         trk != c.end(); ++ trk )
      if ( makeSelection(* trk) ) selected_.push_back( & * trk );
  }
  bool operator()( const reco::Track & t ) const { 
    return  makeSelection( t );
  }

private:
  bool makeSelection( const reco::Track & t ) const { 
    return  t.numberOfValidHits() >= minHit_ &&
      fabs(t.pt()) >= ptMin_ && 
      fabs(t.eta()) >= minRapidity_ && fabs(t.eta()) <= maxRapidity_ && 
      fabs(t.d0()) <= tip_ &&
      fabs(t.dz()) <= lip_ ;
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
