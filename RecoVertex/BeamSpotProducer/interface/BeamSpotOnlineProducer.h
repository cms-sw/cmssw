#ifndef BeamSpotOnlineProducer_BeamSpotOnlineProducer_h
#define BeamSpotOnlineProducer_BeamSpotOnlineProducer_h

/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotOnlineProducer.h,v 1.1 2010/02/25 13:30:35 yumiceva Exp $

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
	virtual void produce(edm::Event& iEvent, const edm::EventSetup& );

  private:
	
	InputTag scalertag_;
	bool changeFrame_;
};

#endif
