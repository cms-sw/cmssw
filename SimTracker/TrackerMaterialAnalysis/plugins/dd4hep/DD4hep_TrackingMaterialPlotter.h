#ifndef DD4hep_TrackingMaterialPlotter_h
#define DD4hep_TrackingMaterialPlotter_h

#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <TH2F.h>
#include <TColor.h>

#include "DD4hep_XHistogram.h"
class MaterialAccountingStep;

class DD4hep_TrackingMaterialPlotter {
public:
  typedef std::pair<double, double> Range;

  DD4hep_TrackingMaterialPlotter(float maxZ, float maxR, float resolution);
  void plotSegmentUnassigned(const MaterialAccountingStep& step);
  void plotSegmentInLayer(const MaterialAccountingStep& step, int layer);

  void normalize(void) { m_tracker.normalize(); }

  void draw(void);

private:
  DD4hep_XHistogram m_tracker;

  std::vector<int> m_color;
  std::vector<int> m_gradient;

  void fill_color();
  unsigned int fill_gradient(const TColor& first, const TColor& last, unsigned int steps = 100, unsigned int index = 0);
  unsigned int fill_gradient(const unsigned int& first,
                             const unsigned int& last,
                             const unsigned int& steps = 100,
                             const unsigned int& index = 0);
};

#endif  //  DD4hep_TrackingMaterialPlotter_h
