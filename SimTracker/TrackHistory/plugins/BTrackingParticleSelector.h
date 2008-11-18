#ifndef BTrackSelection_h
#define BTrackSelection_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackHistory/interface/TrackCategories.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
//#include "CLHEP/HepPDT/ParticleID.hh"

/**
 Selector to select only tracking particles originating from a B-hadron decay.
*/

class BTrackingParticleSelector {

 public:
  // input collection type
  typedef TrackingParticleCollection collection;
  

  // output collection type
  typedef std::vector<const TrackingParticle*> container;

  // iterator over result collection type. 
  typedef container::const_iterator const_iterator;

  // constructor from parameter set configurability
  BTrackingParticleSelector( const edm::ParameterSet & iConfig) : classifier_(iConfig) {};

  // select object from a collection and 
  // possibly event content
  void select( const edm::Handle<collection> & TPCH, const edm::Event & iEvent, const edm::EventSetup & iSetup)
  {
    selected_.clear();

    const collection & tpc = *(TPCH.product());
        
    for(TrackingParticleCollection::size_type i=0; i<tpc.size(); i++){
      
      TrackingParticleRef tp(TPCH, i);

      if( classifier_.evaluate(tp) )
        if( classifier_.is(TrackCategories::Bottom) )
        {
          const TrackingParticle * trap = &(tpc[i]);
          selected_.push_back(trap);
        }  	    	
    }
  }

  // iterators over selected objects: collection begin
  const_iterator begin() const {return selected_.begin();}

  // iterators over selected objects: collection end
  const_iterator end() const {return selected_.end();}

  // true if no object has been selected
  size_t size() const {return selected_.size();}

  //private:

  container selected_;
  TrackCategories classifier_;

};


#endif
