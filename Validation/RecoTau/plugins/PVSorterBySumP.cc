////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
template<typename T1>
class bestPVselector : public edm::EDProducer
{
public:
  // construction/destruction
  bestPVselector(const edm::ParameterSet& iConfig);
  virtual ~bestPVselector();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void endJob();

private:  
  // member data
  edm::InputTag              src_;  
};



////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T1>
bestPVselector<T1>::bestPVselector(const edm::ParameterSet& iConfig)
  : src_(iConfig.getParameter<edm::InputTag>("src"))
{
  produces<std::vector<T1> >();
}


//______________________________________________________________________________
template<typename T1>
bestPVselector<T1>::~bestPVselector(){}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T1>
void bestPVselector<T1>::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{  
  std::auto_ptr<std::vector<T1> > theBestPV(new std::vector<T1 >);
  
  edm::Handle< std::vector<T1> > VertexHandle;
  iEvent.getByLabel(src_,VertexHandle);
  
  if( VertexHandle->size() == 0 ) 
  {
    iEvent.put(theBestPV);
    return ;
  }
  
  typename std::vector<T1>::const_iterator PVit   ;
  typename std::vector<T1>::const_iterator bestPV ;
  
  double bestP4      = 0 ;
  double sumSquarePt = 0 ;

  for (PVit = VertexHandle->begin(); PVit != VertexHandle->end(); ++PVit) {
    sumSquarePt = (PVit -> p4().pt())*(PVit -> p4().pt()) ;
	if( sumSquarePt > bestP4 ){
	  bestP4 = sumSquarePt ;
	  bestPV = PVit        ; 
	}
  }

  theBestPV->push_back( *bestPV );  
  iEvent.put(theBestPV);
  
}

template<typename T1>
void bestPVselector<T1>::endJob()
{
}

typedef bestPVselector<reco::Vertex>      HighestSumP4PrimaryVertexSelector;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HighestSumP4PrimaryVertexSelector);
