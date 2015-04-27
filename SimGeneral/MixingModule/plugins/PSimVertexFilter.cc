#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimGeneral/MixingModule/plugins/PSimVertexFilter.h"

#include <vector>


PSimVertexFilter::PSimVertexFilter(edm::ParameterSet const & config):
    vtxToken_( consumes<CrossingFrame<SimVertex> >( config.getParameter<edm::InputTag> ("simVtxTag") ) )
{
    //register your products
    produces<edm::SimVertexContainer >(simVtxFiltered_);
}


PSimVertexFilter::~PSimVertexFilter()
{}


bool 
PSimVertexFilter::filter(edm::Event& event, const edm::EventSetup& setup)
{
    edm::Handle<CrossingFrame<SimVertex> > genVtxsHandle;
    event.getByToken(vtxToken_, genVtxsHandle);
    const CrossingFrame<SimVertex>* SimVtx = genVtxsHandle.product(); 

    //Create empty output collections
    std::auto_ptr<edm::SimVertexContainer> simVtxFiltered( new edm::SimVertexContainer );
  

    //Select interesting objects
    for(unsigned int iGen=0; iGen<SimVtx->getPileups().size(); ++iGen){
        if(SimVtx->getPileups().at(iGen)->vertexId() == 0)
            simVtxFiltered->push_back(SimVertex(SimVtx->getPileups().at(iGen)->position(), SimVtx->getPileups().at(iGen)->position().t(), iGen)); 
    }

    //Put selected information in the event
    event.put(simVtxFiltered, simVtxFiltered_);
  
    return true;
}


DEFINE_FWK_MODULE(PSimVertexFilter);
