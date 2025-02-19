////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class MuonFromPVSelector : public edm::EDProducer
{
public:
  // construction/destruction
  MuonFromPVSelector(const edm::ParameterSet& iConfig);
  virtual ~MuonFromPVSelector();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void endJob();

private:  
  // member data
  edm::InputTag     srcPart_ ;  
  edm::InputTag     srcPV_   ;
  double            max_dxy_ ;
  double            max_dz_  ;
};



////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
MuonFromPVSelector::MuonFromPVSelector(const edm::ParameterSet& iConfig)
  : srcPart_(iConfig.getParameter<edm::InputTag>("srcMuon"))
  , srcPV_  (iConfig.getParameter<edm::InputTag>("srcVertex"))
  , max_dxy_(iConfig.getParameter<double>("max_dxy"))
  , max_dz_ (iConfig.getParameter<double>("max_dz"))
{
  produces<std::vector<reco::Muon> >();
}


//______________________________________________________________________________
MuonFromPVSelector::~MuonFromPVSelector(){}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void MuonFromPVSelector::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{  
  std::auto_ptr<std::vector<reco::Muon> > goodMuons(new std::vector<reco::Muon >);
  
  edm::Handle< std::vector<reco::Vertex> > VertexHandle;
  iEvent.getByLabel(srcPV_,VertexHandle);

  edm::Handle< std::vector<reco::Muon> > MuonHandle;
  iEvent.getByLabel(srcPart_,MuonHandle);
  
  if( (VertexHandle->size() == 0) || (MuonHandle->size() == 0) ) 
  {
    iEvent.put(goodMuons);
    return ;
  }
  
  
  reco::Vertex PV = VertexHandle->front();   
  //typename std::vector<reco::Muon>::const_iterator MuonIt ;
  std::vector<reco::Muon>::const_iterator MuonIt ;

  for (MuonIt = MuonHandle->begin(); MuonIt != MuonHandle->end(); ++MuonIt) {
    if ( MuonIt->innerTrack().isNonnull()                          &&
         fabs(MuonIt->innerTrack()->dxy(PV.position())) < max_dxy_ &&
         fabs(MuonIt->innerTrack()->dz(PV.position()))  < max_dz_  ){
      goodMuons -> push_back(*MuonIt) ;
    }
  }  
  
  iEvent.put(goodMuons);
  
}

void MuonFromPVSelector::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonFromPVSelector);
