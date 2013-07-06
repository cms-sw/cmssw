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
//#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"

#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
//#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"

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
    //     typedef DigiCollectionFP420<unsigned int, HDigiFP420> DigiColFP420;
    
    explicit DigitizerFP420(const edm::ParameterSet& conf);
    
    virtual ~DigitizerFP420();
    
    //    virtual void produce(PSimHitCollection*, DigiCollectionFP420&);
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
    //     virtual void prodfun(MixCollection<PSimHit>*, DigiCollectionFP420 &);
    //  virtual void prodfun(std::auto_ptr<MixCollection<PSimHit> >*, DigiCollectionFP420 &);
    
    
    //           virtual void prodfun(std::auto_ptr<MixCollection<PSimHit> >&, DigiCollectionFP420 &);
    
  private:
    //  std::vector<PSimHit> theStripHits;
    typedef std::vector<std::string> vstring;
    typedef std::map<unsigned int, std::vector<PSimHit>,std::less<unsigned int> > simhit_map;
    typedef simhit_map::iterator simhit_map_iterator;
    simhit_map SimHitMap;
    
    edm::ParameterSet conf_;
    vstring trackerContainers;
    
    //  HitDigitizerFP420* theHitDigitizerFP420;
    //    FP420DigiMain stripDigitizer_;
    FP420DigiMain* stripDigitizer_;
    FP420NumberingScheme * theFP420NumberingScheme;
    //  FP420DigiMain * theFP420DigiMain;
    int numStrips;    // number of strips in the module
    
    int dn0, sn0, pn0, rn0, verbosity;
    
    
    std::vector<HDigiFP420> collector;
    
    //   DigiCollectionFP420 * output;
    
    
    //   std::vector<edm::DetSet<HDigiFP420> > output;
    
    //      DigiCollectionFP420* poutput;
    
    //  std::map<GeomDetType* , boost::shared_ptr<FP420DigiMain> > theAlgoMap; 
    // std::vector<edm::DetSet<HDigiFP420> > outputfinal;
    //    std::vector<edm::DetSet<HDigiFP420SimLink> > theDigiLinkVector;
    //      std::vector<edm::DetSet<PixelDigi> > theDigiVector;
    
    
    
    
    //    G4ThreeVector bfield(G4ThreeVector);
    //    G4ThreeVector bfield(double, double, double);
    //    G4ThreeVector bfield(float, float, float);
    //    G4ThreeVector bfield();
    
    
  };
}

#endif
