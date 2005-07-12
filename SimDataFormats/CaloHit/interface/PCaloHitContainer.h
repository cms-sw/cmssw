#ifndef _SimDataFormats_SimCaloHit_PCaloHitContainer_h_
#define _SimDataFormats_SimCaloHit_PCaloHitContainer_h_

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "SimDataFormats/SimCaloHit/interface/PCaloHit.h"
#include <vector>
#include <map>
#include <string>

namespace edm {
  class PCaloHitContainer: public EDProduct {
  public:
    typedef std::vector<PCaloHit> PCaloHitSingleContainer;
    typedef std::map<std::string,PCaloHitSingleContainer> PCaloHitMultipleContainer;

    /// insert a digi for a given layer
    void insertHits(std::string name, PCaloHitSingleContainer&);
    void insertHit(std::string name, PCaloHit&);
    
  private:
    PCaloHitMultipleContainer _data;

  };
} // edm


#endif // _SimDataFormats_SimCaloHit_PCaloHitContainer_h_

