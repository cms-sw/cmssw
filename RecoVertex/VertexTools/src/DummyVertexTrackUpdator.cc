#include "RecoVertex/VertexTools/interface/DummyVertexTrackUpdator.h"


RefCountedVertexTrack 
DummyVertexTrackUpdator::update(const CachingVertex & v, 
				RefCountedVertexTrack t) const
{
  return t;
}


DummyVertexTrackUpdator * DummyVertexTrackUpdator::clone() const
{
  return new DummyVertexTrackUpdator(*this);
}
