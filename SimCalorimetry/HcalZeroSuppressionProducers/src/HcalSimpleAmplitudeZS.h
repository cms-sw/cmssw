#ifndef HCALSIMPLEAMPLITUDEZS_H
#define HCALSIMPLEAMPLITUDEZS_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimCalorimetry/HcalZeroSuppressionAlgos/interface/HcalZSAlgoEnergy.h"


/** \class HcalSimpleAmplitudeZS
	
$Date: 2007/03/08 22:13:22 $
$Revision: 1.3 $
\author J. Mans - Minnesota
*/
class HcalSimpleAmplitudeZS : public edm::EDProducer {
public:
  explicit HcalSimpleAmplitudeZS(const edm::ParameterSet& ps);
  virtual ~HcalSimpleAmplitudeZS();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  HcalZSAlgoEnergy algo_;
  std::set<HcalSubdetector> subdets_;
  edm::InputTag inputLabel_;
};

#endif
