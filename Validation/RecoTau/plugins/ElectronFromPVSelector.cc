////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class GsfElectronFromPVSelector : public edm::EDProducer
{
public:
  // construction/destruction
  GsfElectronFromPVSelector(const edm::ParameterSet& iConfig);
  virtual ~GsfElectronFromPVSelector();
  
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
GsfElectronFromPVSelector::GsfElectronFromPVSelector(const edm::ParameterSet& iConfig)
  : srcPart_(iConfig.getParameter<edm::InputTag>("srcElectron"))
  , srcPV_  (iConfig.getParameter<edm::InputTag>("srcVertex"))
  , max_dxy_(iConfig.getParameter<double>("max_dxy"))
  , max_dz_ (iConfig.getParameter<double>("max_dz"))
{
  produces<std::vector<reco::GsfElectron> >();
}


//______________________________________________________________________________
GsfElectronFromPVSelector::~GsfElectronFromPVSelector(){}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void GsfElectronFromPVSelector::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{  
  std::auto_ptr<std::vector<reco::GsfElectron> > goodGsfElectrons(new std::vector<reco::GsfElectron >);
  
  edm::Handle< std::vector<reco::Vertex> > VertexHandle;
  iEvent.getByLabel(srcPV_,VertexHandle);

  edm::Handle< std::vector<reco::GsfElectron> > GsfElectronHandle;
  iEvent.getByLabel(srcPart_,GsfElectronHandle);
  
  if( (VertexHandle->size() == 0) || (GsfElectronHandle->size() == 0) ) 
  {
    iEvent.put(goodGsfElectrons);
    return ;
  }
  
  
  reco::Vertex PV = VertexHandle->front();   
  std::vector<reco::GsfElectron>::const_iterator GsfElectronIt ;
//  typename std::vector<reco::GsfElectron>::const_iterator GsfElectronIt ;

  for (GsfElectronIt = GsfElectronHandle->begin(); GsfElectronIt != GsfElectronHandle->end(); ++GsfElectronIt) {
    
    //int q = GsfElectronIt->gsfTrack()->charge() ;
    
    if ( fabs(GsfElectronIt->gsfTrack()->dxy(PV.position())) < max_dxy_ && 
         fabs(GsfElectronIt->gsfTrack()->dz(PV.position()))  < max_dz_  ) {
    	goodGsfElectrons -> push_back(*GsfElectronIt) ;
    }
  }  
  
  iEvent.put(goodGsfElectrons);
  
}

void GsfElectronFromPVSelector::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfElectronFromPVSelector);
