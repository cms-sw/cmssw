#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"
#include "SimDataFormats/CTPPS/interface/TMultiDimFet.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>

namespace SimDataFormats_CTPPS
{
  struct dictionary
  {
    LHCOpticsApproximator loa;
    LHCApertureApproximator laa;
    TMultiDimFet mdf;

    CTPPSSimProtonTrack csp;
    std::vector<CTPPSSimProtonTrack> vec_csp;
    edm::View<CTPPSSimProtonTrack> v_csp;
    edm::Ptr<CTPPSSimProtonTrack> ptr_csp;
    std::vector< edm::Ptr<CTPPSSimProtonTrack> > vec_ptr_csp;
    edm::Wrapper< std::vector<CTPPSSimProtonTrack> > wrp_vec_csp;
  };
}
