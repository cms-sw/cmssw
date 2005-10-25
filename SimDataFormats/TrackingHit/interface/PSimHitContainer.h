#ifndef SimDataFormats_SimTkHit_PSimHitContainer_H
#define SimDataFormats_SimTkHit_PSimHitContainer_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <vector>
#include <map>
#include <string>

namespace edm {
  class PSimHitContainer {
  public:
      typedef std::vector<PSimHit> PSimHitSingleContainer;
      /// insert a Hit for a given layer
      void insertHits(PSimHitSingleContainer&);
      void insertHit(const PSimHit&);
      void clear();
      // changed by UB
      unsigned int size() const;
      PSimHit operator [] (int i)const  {return _data[i];}
      PSimHitSingleContainer::const_iterator begin () const ;
      PSimHitSingleContainer::const_iterator end () const ;
      PSimHitSingleContainer::iterator begin ();
      PSimHitSingleContainer::iterator end ();


  private:
    PSimHitSingleContainer _data;
  };
} // edm


#endif 

