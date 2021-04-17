#ifndef Validation_MuonCSCDigis_CSCBaseValidation_h
#define Validation_MuonCSCDigis_CSCBaseValidation_h

// user include files

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"

class CSCBaseValidation {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  CSCBaseValidation(const edm::ParameterSet &ps);
  virtual ~CSCBaseValidation() {}
  void setGeometry(const CSCGeometry *geom) { theCSCGeometry = geom; }
  void setSimHitMap(const PSimHitMap *simHitMap) { theSimHitMap = simHitMap; }
  virtual void analyze(const edm::Event &e, const edm::EventSetup &eventSetup) = 0;

protected:
  bool doSim_;
  const CSCLayer *findLayer(int detId) const;
  const PSimHitMap *theSimHitMap;
  const CSCGeometry *theCSCGeometry;
};

#endif
