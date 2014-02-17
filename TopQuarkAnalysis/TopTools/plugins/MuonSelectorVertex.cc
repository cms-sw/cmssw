//
// $Id: MuonSelectorVertex.cc,v 1.1 2012/06/26 16:19:18 vadler Exp $
//


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class MuonSelectorVertex : public edm::EDProducer {

  public:

    explicit MuonSelectorVertex( const edm::ParameterSet & iConfig );
    ~ MuonSelectorVertex() {};
    virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

  private:

    edm::InputTag muonSource_;
    edm::InputTag vertexSource_;
    double        maxDZ_;

};


#include <vector>
#include <memory>
#include <cmath>

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


MuonSelectorVertex::MuonSelectorVertex( const edm::ParameterSet & iConfig )
: muonSource_( iConfig.getParameter< edm::InputTag >( "muonSource" ) )
, vertexSource_( iConfig.getParameter< edm::InputTag >( "vertexSource" ) )
, maxDZ_( iConfig.getParameter< double >( "maxDZ" ) )
{

  produces< std::vector< pat::Muon > >();

}


void MuonSelectorVertex::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{

  edm::Handle< std::vector< pat::Muon > >  muons;
  iEvent.getByLabel( muonSource_, muons );

  edm::Handle< std::vector< reco::Vertex > > vertices;
  iEvent.getByLabel( vertexSource_, vertices );

  std::vector< pat::Muon > * selectedMuons( new std::vector< pat::Muon > );

  if ( vertices->size() > 0 ) {

    for ( unsigned iMuon = 0; iMuon < muons->size(); ++iMuon ) {
      if ( std::fabs( muons->at( iMuon ).vertex().z() - vertices->at( 0 ).z() ) < maxDZ_ ) {
        selectedMuons->push_back( muons->at( iMuon ) );
      }
    }
  }

  std::auto_ptr< std::vector< pat::Muon > > selectedMuonsPtr( selectedMuons );
  iEvent.put( selectedMuonsPtr );

}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( MuonSelectorVertex );
