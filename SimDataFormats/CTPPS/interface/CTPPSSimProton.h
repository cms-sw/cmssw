#ifndef SimDataFormats_CTPPS_CTPPSSimProton_h
#define SimDataFormats_CTPPS_CTPPSSimProton_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

class CTPPSSimProton
{
  public:
    CTPPSSimProton() :
      xi_( 0. ), xi_unc_( 0. ),
      isValid_( false ) {}
    //CTPPSSimProton() {}
    ~CTPPSSimProton() {}

  private:
    Local3DPoint vertex_;
    Local3DVector direction_;

    float xi_;
    float xi_unc_;

    bool isValid_;
};

#endif
