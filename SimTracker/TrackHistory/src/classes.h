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
    edm::Wrapper<std::vector<TrackCategories> > dummy03;
    edm::Wrapper<std::vector<VertexCategories> > dummy04;
};
}

