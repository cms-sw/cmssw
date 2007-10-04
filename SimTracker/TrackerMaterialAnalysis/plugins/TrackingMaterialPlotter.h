#ifndef TrackingMaterialPlotter_h
#define TrackingMaterialPlotter_h

#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <TH2F.h>

#include "XHistogram.h"
class MaterialAccountingStep;

class TrackingMaterialPlotter {
public:
  
  typedef std::pair<double, double> Range;

  TrackingMaterialPlotter( float maxZ, float maxR, float resolution );
  void plotSegmentUnassigned( const MaterialAccountingStep & step );
  void plotSegmentInLayer( const MaterialAccountingStep & step, int layer );

  void normalize( void ) {
    m_tracker.normalize();
  }

  void draw( void );

private:
  XHistogram m_tracker;

  std::vector<int> m_color;
  std::vector<int> m_gradient;

  void fill_color();
  void fill_gradient();

};

#endif // TrackingMaterialPlotter_h
