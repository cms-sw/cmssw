#ifndef SimDataFormats_CTPPS_CTPPSSimProtonTrack_h
#define SimDataFormats_CTPPS_CTPPSSimProtonTrack_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

class CTPPSSimProtonTrack
{
  public:
    CTPPSSimProtonTrack() :
      xi_( 0. ), xi_unc_( 0. ),
      isValid_( false ) {}
    //CTPPSSimProtonTrack() {}
    ~CTPPSSimProtonTrack() {}

    void setVertex( const Local3DPoint& vtx ) { vertex_ = vtx; }
    void setDirection( const Local3DVector& dir ) { direction_ = dir; }
    void setXi( float xi ) { xi_ = xi; }

    void setValid( bool valid=true ) { isValid_ = valid; }

  private:
    Local3DPoint vertex_;
    Local3DVector direction_;

    float xi_;
    float xi_unc_;

    bool isValid_;
};

#endif
