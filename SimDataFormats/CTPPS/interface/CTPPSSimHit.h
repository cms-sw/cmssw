#ifndef SimDataFormats_CTPPS_CTPPSSimHit_h
#define SimDataFormats_CTPPS_CTPPSSimHit_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

class CTPPSSimHit
{
  public:
    CTPPSSimHit() {}
    CTPPSSimHit( const TotemRPDetId& det_id, const Local2DPoint& entry, const Local2DPoint& sigma ) :
      detId_( det_id ), entryPoint_( entry ), entrySigma_( sigma ) {}
    ~CTPPSSimHit() {}

    double getX0() const { return entryPoint_.x(); }
    double getX0Sigma() const { return entrySigma_.x(); }

    double getY0() const { return entryPoint_.y(); }
    double getY0Sigma() const { return entrySigma_.y(); }

    const TotemRPDetId& potId() const { return detId_; }

  private:
    TotemRPDetId detId_;
    Local2DPoint entryPoint_;
    Local2DPoint entrySigma_;
};

#endif
