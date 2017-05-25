#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"
#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"
//#include "SimDataFormats/CTPPS/interface/LHCApertureApproximator.h"
#include "SimDataFormats/CTPPS/interface/TMultiDimFet.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

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

    CTPPSSimHit csh;
    std::vector<CTPPSSimHit> vec_csh;
    edm::View<CTPPSSimHit> v_csh;
    edm::Ptr<CTPPSSimHit> ptr_csh;
    std::vector< edm::Ptr<CTPPSSimHit> > vec_ptr_csh;
    edm::Wrapper< std::vector<CTPPSSimHit> > wrp_vec_csh;
  };
}
