#ifndef BeamSpotOnlineProducer_BeamSpotOnlineProducer_h
#define BeamSpotOnlineProducer_BeamSpotOnlineProducer_h

/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotOnlineProducer.h,v 1.7 2010/09/10 12:13:16 vlimant Exp $

________________________________________________________________**/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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
	
	edm::InputTag scalertag_;
	bool changeFrame_;
	double theMaxZ,theMaxR2,theSetSigmaZ;
	edm::InputTag thel1GtEvmReadoutRecordTag;
};

#endif
