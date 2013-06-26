#ifndef HCALSIMPLEREALISTICZS_H
#define HCALSIMPLEREALISTICZS_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HcalZSAlgoRealistic.h"

//#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
//#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

/** \class HcalSimpleRealisticZS
	
$Date: 2012/10/31 15:34:25 $
$Revision: 1.4 $
\author J. Mans - Minnesota
*/
class HcalRealisticZS : public edm::EDProducer {
public:
  explicit HcalRealisticZS(const edm::ParameterSet& ps);
  virtual ~HcalRealisticZS();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  std::auto_ptr<HcalZSAlgoRealistic> algo_;
  edm::InputTag inputLabel_;
};

#endif
