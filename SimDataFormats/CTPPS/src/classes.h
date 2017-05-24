#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace SimDataFormats_CTPPS
{
  struct dictionary
  {
    CTPPSSimHit csh;
    std::vector<CTPPSSimHit> vec_csh;
    edm::Wrapper< std::vector<CTPPSSimHit> > wrp_vec_csh;
  };
}
