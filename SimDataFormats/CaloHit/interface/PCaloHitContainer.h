#ifndef SimDataFormats_PCaloHitContainer_H
#define SimDataFormats_PCaloHitContainer_H

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include <vector>
#include <string>

namespace edm {

  class PCaloHitContainer {
  public:
    typedef std::vector<PCaloHit> PCaloHitSingleContainer ;

    /// insert a digi for a given layer
    void insertHits (PCaloHitSingleContainer &) ;
    void insertHit (PCaloHit &) ;
    // changed by PG
    void clear () ;

    unsigned int size () const ;
    PCaloHit operator[] (int i) const ; 
 
    PCaloHitSingleContainer::const_iterator begin () const ;
    PCaloHitSingleContainer::const_iterator end () const ;
 
  private:
    PCaloHitSingleContainer m_data ;

  };
} // edm


#endif

