// livio.fano@cern.ch
#ifndef COSMICTIFTRIGFILTER_H
#define COSMICTIFTRIGFILTER_H


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "CLHEP/Vector/ThreeVector.h"

namespace cms{

class CosmicTIFTrigFilter : public edm::EDFilter {
  public:
  CosmicTIFTrigFilter(const edm::ParameterSet& conf);
  virtual ~CosmicTIFTrigFilter() {}
  bool filter(edm::Event & iEvent, edm::EventSetup const& c);
  bool Sci_trig(HepMC::FourVector,  HepMC::FourVector, HepMC::FourVector);
  //  bool Sci_trig(CLHEP::Hep3Vector,  CLHEP::Hep3Vector, CLHEP::Hep3Vector);

 private:
  edm::ParameterSet conf_;

  bool inTK;
  int trigconf;
  int tottrig;
  int trig1, trig2, trig3;
  std::vector<double> trigS1, trigS2, trigS3, trigS4;
  };
}
#endif 
