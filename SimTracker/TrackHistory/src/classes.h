#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimTracker/TrackHistory/interface/TrackCategories.h"
#include "SimTracker/TrackHistory/interface/VertexCategories.h"


namespace
{
struct dictionary
{

    // Dictionaires for Track and Vertex categories

    std::vector<TrackCategories> dummy01;
    std::vector<VertexCategories> dummy02;

    // Dictionaries for SVTagInfoProxy

    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::SecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::Vertex> > >  dummy03;
    edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> > dummy05;

};
}

