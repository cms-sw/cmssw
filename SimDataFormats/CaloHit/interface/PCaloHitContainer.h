#ifndef SimDataFormats_PCaloHitContainer_H
#define SimDataFormats_PCaloHitContainer_H

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include <vector>
#include <string>

namespace edm {
  class PCaloHitContainer: public EDProduct {
  public:
    typedef std::vector<PCaloHit> PCaloHitSingleContainer;

    /// insert a digi for a given layer
    void insertHits(PCaloHitSingleContainer&);
    void insertHit(PCaloHit&);
    
  private:
    PCaloHitSingleContainer _data;

  };
} // edm


#endif

