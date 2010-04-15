#ifndef BeamSpotOnlineProducer_BeamSpotOnlineProducer_h
#define BeamSpotOnlineProducer_BeamSpotOnlineProducer_h

/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotOnlineProducer.h,v 1.2 2010/03/02 17:29:47 yumiceva Exp $

________________________________________________________________**/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

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
	
	InputTag scalertag_;
	bool changeFrame_;
};

#endif
