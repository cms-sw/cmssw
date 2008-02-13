#ifndef BTrackSelection_h
#define BTrackSelection_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"
#include "SimTracker/TrackHistory/interface/TrackOrigin.h"
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


  // default constructor
  BTrackingParticleSelector(): tracer(-2){};

  // constructor from parameter set configurability
  BTrackingParticleSelector( const edm::ParameterSet & ): tracer(-2){};

  // select object from a collection and 
  // possibly event content
  void select( const edm::Handle<collection> & TPCH, const edm::Event & ){
    selected_.clear();

    const collection & tpc = *(TPCH.product());
    
    for(TrackingParticleCollection::size_type i=0; i<tpc.size(); i++){
      
      TrackingParticleRef tp(TPCH, i);

      if(tracer.evaluate(tp)){

	  TrackHistory::GenParticleTrail genParticles(tracer.genParticleTrail());  
	  // Looop over all genParticles
	  bool bselected = false; 
	  for(std::size_t hindex=0; hindex<genParticles.size(); hindex++){
	    int pId = genParticles[hindex]->pdg_id();
	    pId = abs(pId);
	    if(!bselected && ((pId/100)%10 ==5 || (pId/1000)%10 == 5)){
	    //if (!bselected && HepPDT::ParticleID(genParticles[hindex]->pdg_id()).hasBottom()){
	      bselected = true;
	      
	      const TrackingParticle * trap = &(tpc[i]);
	      selected_.push_back(trap);
	    }
	    
	  }
	
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
  TrackOrigin tracer;

};


#endif
