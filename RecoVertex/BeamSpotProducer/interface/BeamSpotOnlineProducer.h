#ifndef BeamSpotOnlineProducer_BeamSpotOnlineProducer_h
#define BeamSpotOnlineProducer_BeamSpotOnlineProducer_h

/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"


class BeamSpotOnlineProducer: public edm::stream::EDProducer<> {

  public:
	typedef std::vector<edm::ParameterSet> Parameters;

	/// constructor
	explicit BeamSpotOnlineProducer(const edm::ParameterSet& iConf);
	/// destructor
	~BeamSpotOnlineProducer() override;
	
	/// produce a beam spot class
	void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:

	const bool changeFrame_;
	const double theMaxZ,theSetSigmaZ;
	double theMaxR2;
	const edm::EDGetTokenT<BeamSpotOnlineCollection> scalerToken_;
	const edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> l1GtEvmReadoutRecordToken_;

	const unsigned int theBeamShoutMode;
};

#endif
