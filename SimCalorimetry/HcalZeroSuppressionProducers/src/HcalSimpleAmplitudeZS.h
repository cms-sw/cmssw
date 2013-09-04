#ifndef HCALSIMPLEAMPLITUDEZS_H
#define HCALSIMPLEAMPLITUDEZS_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HcalZSAlgoEnergy.h"

#include <string>

/** \class HcalSimpleAmplitudeZS
	
$Date: 2013/04/24 10:22:20 $
$Revision: 1.5 $
\author J. Mans - Minnesota
*/
class HcalSimpleAmplitudeZS : public edm::EDProducer {
public:
  explicit HcalSimpleAmplitudeZS(const edm::ParameterSet& ps);
  virtual ~HcalSimpleAmplitudeZS();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  std::auto_ptr<HcalZSAlgoEnergy> hbhe_,ho_,hf_,hbheUpgrade_,hfUpgrade_;
  std::string inputLabel_;
};

#endif
