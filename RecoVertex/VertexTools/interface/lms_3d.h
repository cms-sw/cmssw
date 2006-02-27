#ifndef LMS_3D_ICC
#define LMS_3D_ICC

#include "Geometry/Vector/interface/GlobalPoint.h"
#include <vector>

/// least median of squares in three dimensions,
/// doing every dimension separately
GlobalPoint lms_3d ( std::vector<GlobalPoint> values );

#endif
