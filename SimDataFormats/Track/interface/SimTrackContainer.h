#ifndef SimTrackContainer_H
#define SimTrackContainer_H

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>
/// why defined in namespace edm:: ?? (L.L.)
namespace edm {
  typedef std::vector<SimTrack> SimTrackContainer;
}
typedef edm::Ref<edm::SimTrackContainer> SimTrackRef;
typedef edm::RefProd<edm::SimTrackContainer> SimTrackRefProd;
typedef edm::RefVector<edm::SimTrackContainer> SimTrackRefVector;

#endif
