//
//


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


class MuonSelectorVertex : public edm::EDProducer {

  public:

    explicit MuonSelectorVertex( const edm::ParameterSet & iConfig );
    ~ MuonSelectorVertex() {};
    virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) override;

  private:

    edm::EDGetTokenT< std::vector< pat::Muon > > muonSource_;
    edm::EDGetTokenT< std::vector< reco::Vertex > > vertexSource_;
    double        maxDZ_;

};


#include <vector>
#include <memory>
#include <cmath>


MuonSelectorVertex::MuonSelectorVertex( const edm::ParameterSet & iConfig )
: muonSource_( consumes< std::vector< pat::Muon > >( iConfig.getParameter< edm::InputTag >( "muonSource" ) ) )
, vertexSource_( consumes< std::vector< reco::Vertex > >( iConfig.getParameter< edm::InputTag >( "vertexSource" ) ) )
, maxDZ_( iConfig.getParameter< double >( "maxDZ" ) )
{

  produces< std::vector< pat::Muon > >();

}


void MuonSelectorVertex::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{

  edm::Handle< std::vector< pat::Muon > >  muons;
  iEvent.getByToken( muonSource_, muons );

  edm::Handle< std::vector< reco::Vertex > > vertices;
  iEvent.getByToken( vertexSource_, vertices );

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
