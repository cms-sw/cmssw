#ifndef BeamSpotProducer_BeamSpotProducer_h
#define BeamSpotProducer_BeamSpotProducer_h

/**_________________________________________________________________
   class:   BeamSpotProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotProducer.h,v 1.1 2007/05/09 20:35:49 yumiceva Exp $

________________________________________________________________**/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


class BeamSpotProducer: public edm::EDProducer {

  public:
	typedef std::vector<edm::ParameterSet> Parameters;

	/// constructor
	explicit BeamSpotProducer(const edm::ParameterSet& iConf);
	/// destructor
	~BeamSpotProducer();
	
	/// produce a beam spot class
	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:
	
};

#endif
