#ifndef BeamSpotProducer_BeamSpotFromDB_h
#define BeamSpotProducer_BeamSpotFromDB_h

/**_________________________________________________________________
   class:   BeamSpotFromDB.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotFromDB.h,v 1.2 2009/12/18 20:45:07 wmtan Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class BeamSpotFromDB : public edm::EDAnalyzer {
 public:
  explicit BeamSpotFromDB(const edm::ParameterSet&);
  ~BeamSpotFromDB();

 private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


};

#endif
