#ifndef TrackingMaterialPlotter_h
#define TrackingMaterialPlotter_h

#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <TH2F.h>
#include <TColor.h>

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

  void reset( void ) {
    m_tracker.reset();
  }

  void draw( void );

private:
  XHistogram m_tracker;

  std::vector<int> m_color;
  std::vector<int> m_gradient;

  void fill_color();
  unsigned int fill_gradient(const TColor & first, const TColor & last, unsigned int steps = 100, unsigned int index = 0);
  unsigned int fill_gradient(unsigned int first, unsigned int last, unsigned int steps = 100, unsigned int index = 0);

};

#endif // TrackingMaterialPlotter_h
