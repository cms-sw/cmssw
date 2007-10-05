#ifndef MaterialAccountingDetector_h
#define MaterialAccountingDetector_h

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// CMSSW
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"

// struct to keep material accounting and geometrical informations on a per-detector basis, along a track
class MaterialAccountingDetector {
  friend class MaterialAccountingTrack;
  friend class TrackingMaterialAnalyser;

public:
  MaterialAccountingDetector( void ) :
    m_position         (),
    m_curvilinearIn    ( 0. ),
    m_curvilinearOut   ( 0. ),
    m_cosThetaIn       ( 0. ),
    m_cosThetaOut      ( 0. ),
    m_accounting       ()
  { }
  
  void clear( void ) {
    m_position         = GlobalPoint(),
    m_curvilinearIn    = 0.;
    m_curvilinearOut   = 0.;
    m_cosThetaIn       = 0.;
    m_cosThetaOut      = 0.;
    m_accounting.clear();
  }

  const GlobalPoint & position() const {
    return m_position;
  }

  const MaterialAccountingStep & material() const {
    return m_accounting;
  }

  // step holds the length and material infos for a step
  // begin and end are the curviliniear coordinates
  void account( const MaterialAccountingStep & step, double begin, double end )
  {
    if (end <= m_curvilinearIn)
      // step before detector
      m_accounting += m_cosThetaIn * step;
    else if (begin >= m_curvilinearOut)
      // step after detector
       m_accounting += m_cosThetaOut * step;
    else
      // step inside detector
      m_accounting += (m_cosThetaIn + m_cosThetaOut ) / 2. * step;
  }
  
private:  
  GlobalPoint m_position;                       // roughly the center of the detector
  double m_curvilinearIn;                       // beginning of detector coordinate along the track 
  double m_curvilinearOut;                      // end of detector coordinate along the track 
  double m_cosThetaIn;
  double m_cosThetaOut;
  MaterialAccountingStep m_accounting;
};

#endif // MaterialAccountingDetector_h
