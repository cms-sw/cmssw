#include "DataFormats/Common/interface/Ref.h"
#include <vector>
namespace l1slhc {
class L1CaloRegion;

  typedef std::vector<L1CaloRegion> L1CaloRegionCollection;
  typedef edm::Ref<L1CaloRegionCollection> L1CaloRegionRef;
  typedef std::vector<L1CaloRegionRef> L1CaloRegionRefVector;
}
