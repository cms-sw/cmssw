#ifndef SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H
#define SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "CondFormats/DataRecord/interface/EcalTPParametersRcd.h"

//
// class declaration
//

class EcalTrigPrimESProducer : public edm::ESProducer {
 public:
  EcalTrigPrimESProducer(const edm::ParameterSet&);
  ~EcalTrigPrimESProducer();

  typedef std::auto_ptr<EcalTPParameters> ReturnType;

  ReturnType produce(const EcalTPParametersRcd&);

  // these constants are put here in order to stay independent from geometry
  // this makes us be fast in case of barrelonly

  static const int MIN_TCC_EB;
  static const int MAX_TCC_EB;
  static const int MIN_TCC_EE_PLUS;
  static const int MAX_TCC_EE_PLUS;
  static const int MIN_TCC_EE_MINUS;
  static const int MAX_TCC_EE_MINUS;
  static const int MIN_TT_EB;
  static const int MAX_TT_EB;
  static const int MIN_TT_EE; 
  static const int MAX_TT_EE; //This is a maximum from outer (=16) and inner (=24 without 4 virtual ones)
  static const int MIN_STRIP_EB;
  static const int MAX_STRIP_EB;
  static const int MIN_STRIP_EE;
  static const int MAX_STRIP_EE;
  static const int MIN_XTAL_EB;
  static const int MAX_XTAL_EB;
  static const int MIN_XTAL_EE;
  static const int MAX_XTAL_EE;

 private:

  void parseTextFile(EcalTPParameters &);

  std::vector<int> getRange(int subdet, int smNb, int towerNbInSm, int stripNbInTower=0, int xtalNbInStrip=0) ;

  // ----------member data ---------------------------
  std::string dbFilenameEB_;
  std::string dbFilenameEE_;

};

#endif
