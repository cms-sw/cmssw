#ifndef CSCRecHitValidation_h
#define CSCRecHitValidation_h

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Validation/CSCRecHits/src/CSCRecHit2DValidation.h"
#include "Validation/CSCRecHits/src/CSCSegmentValidation.h"



class CSCRecHitValidation : public edm::EDAnalyzer {
public:
  explicit CSCRecHitValidation(const edm::ParameterSet&);
  ~CSCRecHitValidation();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(){} 
  virtual void endJob() ;
 

 private:

  DQMStore* dbe_;
  std::string theOutputFile;
  PSimHitMap theSimHitMap;
  const CSCGeometry * theCSCGeometry;

  CSCRecHit2DValidation * the2DValidation;
  CSCSegmentValidation * theSegmentValidation;
};

#endif

