////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

#include "DataFormats/Common/interface/View.h"

#include <memory>
#include <vector>
#include <sstream>

//#include "Validation/RecoTau/interface/prettyPrint.h" debugging putpose

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class CollectionFromZLegProducer : public edm::EDProducer
{
public:
  // construction/destruction
  CollectionFromZLegProducer(const edm::ParameterSet& iConfig);
  virtual ~CollectionFromZLegProducer();
  
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup) override;

private:  
  // member data
  edm::EDGetTokenT< std::vector<reco::CompositeCandidate> >   v_RecoCompositeCandidateToken_;
  std::string                                                 OutputCollection_ ;
  
};



////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
CollectionFromZLegProducer::CollectionFromZLegProducer(const edm::ParameterSet& iConfig)
  : v_RecoCompositeCandidateToken_( consumes< std::vector<reco::CompositeCandidate> >( iConfig.getParameter<edm::InputTag>( "ZCandidateCollection" ) ) )
{
  produces<std::vector<reco::CompositeCandidate> >("theTagLeg"  );
  produces<std::vector<reco::CompositeCandidate> >("theProbeLeg");
}


//______________________________________________________________________________
CollectionFromZLegProducer::~CollectionFromZLegProducer()
{
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void CollectionFromZLegProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{  
  std::auto_ptr<std::vector<reco::CompositeCandidate> > theTagLeg(new std::vector<reco::CompositeCandidate>) ;	     
  std::auto_ptr<std::vector<reco::CompositeCandidate> > theProbeLeg(new std::vector<reco::CompositeCandidate>) ;	     
  
  edm::Handle< std::vector<reco::CompositeCandidate> > theZHandle;
  iEvent.getByToken( v_RecoCompositeCandidateToken_,theZHandle );
  
  // this is specific for our 'tag and probe'
  
  for (std::vector<reco::CompositeCandidate>::const_iterator Zit  = theZHandle->begin() ; 
                                                             Zit != theZHandle->end()   ; 
                                                             ++Zit                      )
  {
	int c = 0;
	
	for(reco::CompositeCandidate::const_iterator Daug =(*Zit).begin(); 
                                                 Daug!=(*Zit).end()  ; 
                                                 ++Daug              )
	{
	  if (c == 0){
	    reco::CompositeCandidate candT(*Daug) ;
	    theTagLeg->push_back(candT) ;
	  }
	  if (c == 1){
	    reco::CompositeCandidate candP(*Daug) ;
	    theProbeLeg->push_back(candP) ;
	  }
	  c++ ;
	}
  } 
  iEvent.put(theTagLeg  , "theTagLeg"   ) ;
  iEvent.put(theProbeLeg, "theProbeLeg" ) ;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CollectionFromZLegProducer);
