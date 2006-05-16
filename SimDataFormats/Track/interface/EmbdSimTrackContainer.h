#ifndef EmbdSimTrackContainer_H
#define EmbdSimTrackContainer_H

#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>
/// why defined in namespace edm:: ?? (L.L.)
namespace edm {
  typedef std::vector<EmbdSimTrack> EmbdSimTrackContainer;
}
typedef edm::Ref<edm::EmbdSimTrackContainer> EmbdSimTrackRef;
typedef edm::RefProd<edm::EmbdSimTrackContainer> EmbdSimTrackRefProd;
typedef edm::RefVector<edm::EmbdSimTrackContainer> EmbdSimTrackRefVector;

#endif
