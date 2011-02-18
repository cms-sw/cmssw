#ifndef DigitizerFP420_h
#define DigitizerFP420_h

///////////////////////////////////////////////////////////////////////
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//////////////////
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"
#include "SimRomanPot/SimFP420/interface/HPS240DigiMain.h"

#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"

#include <CLHEP/Vector/ThreeVector.h>
#include <string>
#include<vector>
#include <iostream>
#include <map>

namespace cms
{
  class DigitizerFP420: public edm::EDProducer
  {
  public:
    
    explicit DigitizerFP420(const edm::ParameterSet& conf);
    
    virtual ~DigitizerFP420();
    
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    typedef std::vector<std::string> vstring;
    typedef std::map<unsigned int, std::vector<PSimHit>,std::less<unsigned int> > simhit_map;
    typedef simhit_map::iterator simhit_map_iterator;
    simhit_map SimHitMap;
    
    edm::ParameterSet conf_;
    vstring trackerContainers;
    
    FP420DigiMain* stripDigitizer_;
    HPS240DigiMain* HPS240Digitizer_;

    FP420NumberingScheme * theFP420NumberingScheme;
    //  int numStrips;    // number of strips in the module
    
    int verbosity;
    int dn0, sn0, pn0, rn0;
    int dh0, sh0, ph0, rh0;
    
    
    std::vector<HDigiFP420> collector;
    
  };
}

#endif
