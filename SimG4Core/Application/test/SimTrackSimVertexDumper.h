#ifndef SimTrackSimVertexDumper_H
#define SimTrackSimVertexDumper_H
// 
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <vector>

class SimTrackSimVertexDumper : public edm::EDAnalyzer{
   public:
  explicit SimTrackSimVertexDumper( const edm::ParameterSet& );
  virtual ~SimTrackSimVertexDumper() {};
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginJob( const edm::EventSetup& ){};
  virtual void endJob(){};
   private:
  std::string HepMCLabel;
  std::string SimTkLabel;
  std::string SimVtxLabel;
  bool dumpHepMC;

};

#endif
