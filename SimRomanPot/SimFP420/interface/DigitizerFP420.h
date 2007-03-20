#ifndef DigitizerFP420_h
#define DigitizerFP420_h

///////////////////////////////////////////////////////////////////////
/*
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
*/
///////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"

#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"
#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"
#include "SimRomanPot/SimFP420/interface/ClusterFP420.h"

#include <CLHEP/Vector/ThreeVector.h>
#include <string>
#include<vector>
#include <iostream>


//  class FP420NumberingScheme;
//  class FP420DigiMain;

//namespace fp420
//{
  class DigitizerFP420: public SimWatcher
  {
  public:

    explicit DigitizerFP420(const edm::ParameterSet& conf);

    virtual ~DigitizerFP420();

    //    virtual void produce(FP420G4HitCollection*, DigiCollectionFP420&);
    virtual void produce(FP420G4HitCollection*, DigiCollectionFP420 &);

  private:
    edm::ParameterSet conf_;
//  HitDigitizerFP420* theHitDigitizerFP420;
//    FP420DigiMain stripDigitizer_;
    FP420DigiMain* stripDigitizer_;

    std::vector<FP420G4Hit> theStripHits;
    typedef std::map<unsigned int, std::vector<FP420G4Hit>,std::less<unsigned int> > simhit_map;
    typedef simhit_map::iterator simhit_map_iterator;
    simhit_map SimHitMap;

    std::vector<HDigiFP420> collector;
    FP420NumberingScheme * theFP420NumberingScheme;
    FP420DigiMain * theFP420DigiMain;
    int numStrips;    // number of strips in the module

    std::vector<ClusterFP420> scollector;

    int sn0, pn0, verbosity;

    //    G4ThreeVector bfield(G4ThreeVector);
    //    G4ThreeVector bfield(double, double, double);
    //    G4ThreeVector bfield(float, float, float);
    //    G4ThreeVector bfield();


  };


#endif
