// -*- C++ -*-
//
// Package:    SVTagInfoVertexAdapter
// Class:      SVTagInfoVertexAdapter
//
/**\class SVTagInfoVertexAdapter SVTagInfoVertexAdapter.cc SimTracker/SVTagInfoVertexAdapter/src/SVTagInfoVertexAdapter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Victor Bazterra, Maria Aldaya
//         Created:  Tue Feb 24 09:42:18 CST 2009
// $Id: SVTagInfoVertexAdapter.cc,v 1.1 2009/02/24 18:43:34 bazterra Exp $
//
//

// system include files
#include <memory>

// user include files
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
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

class SVTagInfoVertexAdapter : public edm::EDProducer
{
public:
    explicit SVTagInfoVertexAdapter(const edm::ParameterSet&);

private:
    virtual void produce(edm::Event&, const edm::EventSetup&);

    edm::InputTag svTagInfoCollection_;
};

SVTagInfoVertexAdapter::SVTagInfoVertexAdapter(const edm::ParameterSet& config)
{
    // Get the cfg parameter
    svTagInfoCollection_ = config.getUntrackedParameter<edm::InputTag> ( "svTagInfoProducer" );

    // Declare the type of object to be produced.
    produces<reco::VertexCollection>();
}

void SVTagInfoVertexAdapter::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // Vertex collection
    edm::Handle<reco::SecondaryVertexTagInfoCollection> svTagInfoCollection;
    event.getByLabel(svTagInfoCollection_, svTagInfoCollection);

    // Auto pointer to the collection to be added to the event
    std::auto_ptr<reco::VertexCollection> results (new reco::VertexCollection);

    // Loop over SecondaryVertexTagInfo collection
    for (
        reco::SecondaryVertexTagInfoCollection::const_iterator svTagInfo = svTagInfoCollection->begin();
        svTagInfo != svTagInfoCollection->end();
        ++svTagInfo
    )
    {
        // Loop over the vertexes and add them to the new collection
        for (unsigned int index = 0; index < svTagInfo->nVertices(); ++index)
            results->push_back( svTagInfo->secondaryVertex(index) );
    }

    // Adding the collection to the event
    event.put(results);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SVTagInfoVertexAdapter);
