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
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup) override;

private:  
  // member data
  double                                               max_dxy_               ;
  double                                               max_dz_                ;
  edm::EDGetTokenT< std::vector<reco::Vertex> >        v_recoVertexToken_     ;
  edm::EDGetTokenT< std::vector<reco::GsfElectron> >   v_recoGsfElectronToken_;
};



////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
GsfElectronFromPVSelector::GsfElectronFromPVSelector(const edm::ParameterSet& iConfig)
  : max_dxy_               ( iConfig.getParameter<double>( "max_dxy" ) )
  , max_dz_                ( iConfig.getParameter<double>( "max_dz" ) )
  , v_recoVertexToken_     ( consumes< std::vector<reco::Vertex> >( iConfig.getParameter<edm::InputTag>( "srcVertex" ) ) )
  , v_recoGsfElectronToken_( consumes< std::vector<reco::GsfElectron> >( iConfig.getParameter<edm::InputTag>( "srcElectron" ) ) )
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
  iEvent.getByToken( v_recoVertexToken_, VertexHandle );

  edm::Handle< std::vector<reco::GsfElectron> > GsfElectronHandle;
  iEvent.getByToken( v_recoGsfElectronToken_, GsfElectronHandle );
  
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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfElectronFromPVSelector);
