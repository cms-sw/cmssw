#ifndef TrackParticleSelection_h
#define TrackParticleSelection_h

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/Common/interface/RefTraits.h"

#include "SimTracker/TrackHistory/interface/TrackCategories.h"

/**
 Selector to select only tracking particles originating from a B-hadron decay.
*/

#include "DataFormats/Common/interface/RefItem.h"

template <typename Collection, TrackCategories::Category Category>
class TrackParticleSelector {

 public:
  
  // input collection type  typedef Collection collection;  
  // type of the collection elements
  typedef typename Collection::value_type type;
   
  // output collection type
  typedef std::vector<const type *> container;

  // iterator over result collection type. 
  typedef typename container::const_iterator const_iterator;

  // constructor from parameter set configurability
  TrackParticleSelector( const edm::ParameterSet & pset ) : classifier_(pset) {}

  // select object from a collection and 
  // possibly event content
  void select( const edm::Handle<collection> & TPCH, const edm::Event & iEvent, const edm::EventSetup &iSetup)
  {
    selected_.clear();

    const collection & tpc = *(TPCH.product());

    classifier_.newEvent(iEvent, iSetup);
            
    for(typename collection::size_type i=0; i<tpc.size(); i++)
    {  
      edm::Ref<Collection> tp(TPCH, i);

      if( classifier_.evaluate(tp) )
        if( classifier_.is(Category) )
        {
          const type * trap = &(tpc[i]);
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

 private:

  TrackCategories classifier_;

};


#endif
