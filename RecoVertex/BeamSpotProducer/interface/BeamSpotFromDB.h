#ifndef BeamSpotProducer_BeamSpotFromDB_h
#define BeamSpotProducer_BeamSpotFromDB_h

/**_________________________________________________________________
   class:   BeamSpotFromDB.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

class BeamSpotFromDB : public edm::EDAnalyzer {
public:
  explicit BeamSpotFromDB(const edm::ParameterSet&);
  ~BeamSpotFromDB() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> m_beamToken;
};

#endif
