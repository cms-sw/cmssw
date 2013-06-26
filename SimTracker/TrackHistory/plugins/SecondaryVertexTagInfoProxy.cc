
// system include files
#include <memory>

// user include files
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

class SecondaryVertexTagInfoProxy : public edm::EDProducer
{
public:

    explicit SecondaryVertexTagInfoProxy(const edm::ParameterSet&);

private:

    virtual void produce(edm::Event&, const edm::EventSetup&);

    edm::InputTag svTagInfoCollection_;
};


SecondaryVertexTagInfoProxy::SecondaryVertexTagInfoProxy(const edm::ParameterSet& config)
{
    // Get the cfg parameter
    svTagInfoCollection_ = config.getUntrackedParameter<edm::InputTag> ( "svTagInfoProducer" );

    // Declare the type of objects to be produced.
    produces<reco::VertexCollection>();
    produces<edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> > >();
}


void SecondaryVertexTagInfoProxy::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // Vertex collection
    edm::Handle<reco::SecondaryVertexTagInfoCollection> svTagInfoCollection;
    event.getByLabel(svTagInfoCollection_, svTagInfoCollection);

    // Auto pointers to the collection to be added to the event
    std::auto_ptr<reco::VertexCollection> proxy (new reco::VertexCollection);
    std::auto_ptr<edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> > >
    assoc (new edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> >);

    // Get a reference before to put in the event
    reco::VertexRefProd vertexRefProd = event.getRefBeforePut<reco::VertexCollection>();

    // General index
    std::size_t index = 0;

    // Loop over SecondaryVertexTagInfo collection
    for (std::size_t svIndex = 0; svIndex < svTagInfoCollection->size(); ++svIndex)
    {
        // Reference to svTagInfo
        reco::SecondaryVertexTagInfoRef svTagInfo(svTagInfoCollection, svIndex);

        // Loop over the vertexes and add them to the new collection
        for (unsigned int vIndex = 0; vIndex < svTagInfo->nVertices(); ++vIndex)
        {
            proxy->push_back(svTagInfo->secondaryVertex(vIndex));
            assoc->insert(svTagInfo, reco::VertexRef(vertexRefProd, index));
            ++index;
        }
    }

    // Adding the collection to the event
    event.put(proxy);
    event.put(assoc);
}


//define this as a plug-in
DEFINE_FWK_MODULE(SecondaryVertexTagInfoProxy);
