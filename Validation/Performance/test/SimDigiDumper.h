#ifndef SimDigiDumper_H
#define SimDigiDumper_H
// 
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

class SimDigiDumper : public edm::EDAnalyzer{
 public:
  explicit SimDigiDumper( const edm::ParameterSet& );
  virtual ~SimDigiDumper() {};
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginJob(){};
  virtual void endJob(){};
 private:
  
  edm::InputTag ECalEBSrc_;
  edm::InputTag ECalEESrc_;
  edm::InputTag ECalESSrc_;

  edm::InputTag HCalDigi_;

  edm::InputTag ZdcDigi_;

  edm::InputTag SiStripSrc_;

  edm::InputTag SiPxlSrc_;
  
  edm::InputTag MuDTSrc_;

  edm::InputTag MuCSCStripSrc_;

  edm::InputTag MuCSCWireSrc_;

  edm::InputTag MuRPCSrc_;

  static const int sdSiTIB          = 3;
  static const int sdSiTID          = 4;
  static const int sdSiTOB          = 5;
  static const int sdSiTEC          = 6;
  static const int sdPxlBrl         = 1;
  static const int sdPxlFwd         = 2;


};

#endif
