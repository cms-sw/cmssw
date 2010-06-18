#ifndef BeamSpotProducer_AlcaBeamSpotFromDB_h
#define BeamSpotProducer_AlcaBeamSpotFromDB_h

/**_________________________________________________________________
   class:   AlcaBeamSpotFromDB.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: AlcaBeamSpotFromDB.h,v 1.2 2009/12/18 20:45:07 wmtan Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

class AlcaBeamSpotFromDB : public edm::EDProducer {
 public:
  explicit AlcaBeamSpotFromDB(const edm::ParameterSet&);
  ~AlcaBeamSpotFromDB();

 private:
  virtual void beginJob() ;
  virtual void beginLuminosityBlock(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
  virtual void endLuminosityBlock  (edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
  virtual void produce             (edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob() ;


};

#endif
