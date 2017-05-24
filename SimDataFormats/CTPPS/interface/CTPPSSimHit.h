#ifndef SimDataFormats_CTPPS_CTPPSSimHit_h
#define SimDataFormats_CTPPS_CTPPSSimHit_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class CTPPSSimHit
{
  public:
    CTPPSSimHit() : pdgId_( 0 ) {}
    CTPPSSimHit( const Local3DPoint& entry, int pdg_id ) :
      entryPoint_( entry ), pdgId_( pdg_id ) {}
    ~CTPPSSimHit() {}

  private:
    Local3DPoint entryPoint_;
    int pdgId_;
};

#endif
