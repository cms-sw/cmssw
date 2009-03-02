#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimTracker/TrackHistory/interface/TrackCategories.h"
#include "SimTracker/TrackHistory/interface/VertexCategories.h"


namespace
{
namespace
{

// Wrapper for Track and Vertex categories
edm::Wrapper<std::vector<TrackCategories> > dummy01;
edm::Wrapper<std::vector<VertexCategories> > dummy02;

// Instances for SVTagInfoProxy
edm::helpers::KeyVal<edm::RefProd<std::vector<reco::SecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::Vertex> > >  dummy03;
edm::Wrapper<edm::helpers::KeyVal<edm::RefProd<std::vector<reco::SecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::Vertex> > > > dummy04;
edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> > dummy05;
edm::Wrapper<edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> > > dummy06;

}
}

