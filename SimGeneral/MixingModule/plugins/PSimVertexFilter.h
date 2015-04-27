#ifndef _PSimVertexFilter_H
#define _PSimVertexFilter_H

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"


//
// class decleration
//

class PSimVertexFilter : public edm::EDFilter{
public:
    explicit PSimVertexFilter(const edm::ParameterSet&);
    ~PSimVertexFilter();

    virtual bool filter(edm::Event& event, const edm::EventSetup& setup);

private:
    const edm::EDGetTokenT<CrossingFrame<SimVertex> > vtxToken_;
    const std::string simVtxFiltered_;
};

#endif
