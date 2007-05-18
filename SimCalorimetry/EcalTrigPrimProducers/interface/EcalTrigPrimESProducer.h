#ifndef SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H
#define SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

/* #include "FWCore/Framework/interface/ESHandle.h" */
/* #include "FWCore/ParameterSet/interface/ParameterSet.h" */

/* #include "CondFormats/L1TObjects/interface/EcalTPParameters.h" */
/* #include "CondFormats/DataRecord/interface/EcalTPParametersRcd.h" */

//
// class declaration
//

class EcalTrigPrimESProducer : public edm::ESProducer {
 public:
  //head  EcalTrigPrimESProducer(const edm::ParameterSet&);
  //131
  EcalTrigPrimESProducer(const edm::ParameterSet&) {;}
  ~EcalTrigPrimESProducer();

/*   typedef std::auto_ptr<EcalTPParameters> ReturnType; */

/*   ReturnType produce(const EcalTPParametersRcd&); */

/*  private: */

/*   void parseTextFile(EcalTPParameters &); */

/*   std::vector<int> getRange(int smNb, int towerNbInSm, int stripNbInTower=0, int xtalNbInStrip=0) ; */

/*   // ----------member data --------------------------- */
/*   std::string dbFilenameEB_; */
/*   std::string dbFilenameEE_; */

};

#endif
