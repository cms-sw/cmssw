/****************************************************************************
 *
 * This is a part of CTPPS offline software
 * Authors:
 *   Leszek Grzanka
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

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
    CTPPSSimProtonTrack( const Local3DPoint& vtx, const Local3DVector& dir, float xi, float xi_unc=0. ) :
      vertex_( vtx ), direction_( dir ), xi_( xi ), xi_unc_( xi_unc ),
      isValid_( true ) {}
    ~CTPPSSimProtonTrack() {}

    void setVertex( const Local3DPoint& vtx ) { vertex_ = vtx; }
    const Local3DPoint& vertex() const { return vertex_; }

    void setDirection( const Local3DVector& dir ) { direction_ = dir; }
    const Local3DVector& direction() const { return direction_; }

    void setXi( float xi ) { xi_ = xi; }
    float xi() const { return xi_; }

    void setValid( bool valid=true ) { isValid_ = valid; }
    bool valid() const { return isValid_; }

  private:
    Local3DPoint vertex_;
    Local3DVector direction_;

    float xi_;
    float xi_unc_;

    bool isValid_;
};

#endif
