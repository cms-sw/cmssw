#ifndef BeamSpotOnlineProducer_BeamSpotOnlineProducer_h
#define BeamSpotOnlineProducer_BeamSpotOnlineProducer_h

/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"


class BeamSpotOnlineProducer: public edm::EDProducer {

  public:
	typedef std::vector<edm::ParameterSet> Parameters;

	/// constructor
	explicit BeamSpotOnlineProducer(const edm::ParameterSet& iConf);
	/// destructor
	~BeamSpotOnlineProducer();
	
	/// produce a beam spot class
	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:

	bool changeFrame_;
	double theMaxZ,theMaxR2,theSetSigmaZ;
	edm::EDGetTokenT<BeamSpotOnlineCollection> scalerToken_;
	edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> l1GtEvmReadoutRecordToken_;
};

#endif
