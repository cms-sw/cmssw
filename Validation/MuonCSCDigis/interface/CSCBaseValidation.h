#ifndef CSCBaseValidation_h
#define CSCBaseValidation_h

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Validation/MuonCSCDigis/interface/CSCPSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"


class CSCBaseValidation {
public:
  CSCBaseValidation(DQMStore* dbe,
                    const edm::InputTag & inputTag);
  virtual ~CSCBaseValidation() {}

  void setGeometry(const CSCGeometry * geom) {theCSCGeometry = geom;}
  void setSimHitMap(const  CSCPSimHitMap * simHitMap) {theSimHitMap = simHitMap;}

  virtual void analyze(const edm::Event&e, const edm::EventSetup& eventSetup) = 0;

protected:
  const CSCLayer * findLayer(int detId) const;

  DQMStore* dbe_;
  edm::InputTag theInputTag;
  const CSCPSimHitMap * theSimHitMap;
  const CSCGeometry * theCSCGeometry;
};

#endif

