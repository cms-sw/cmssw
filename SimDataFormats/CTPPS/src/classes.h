#include "SimDataFormats/CTPPS/interface/CTPPSSimProton.h"
#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace SimDataFormats_CTPPS
{
  struct dictionary
  {
    CTPPSSimProton csp;
    std::vector<CTPPSSimProton> vec_csp;
    edm::Wrapper< std::vector<CTPPSSimProton> > wrp_vec_csp;

    CTPPSSimHit csh;
    std::vector<CTPPSSimHit> vec_csh;
    edm::Wrapper< std::vector<CTPPSSimHit> > wrp_vec_csh;
  };
}
