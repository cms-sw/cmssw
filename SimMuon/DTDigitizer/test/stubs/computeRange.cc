#include "computeRange.h"

#include <cmath>

float computeRange(float energy, float density) {
  float range = 0;
  range = ((5.37e2) * energy * (1 - (0.9815) / (1 + (energy * 3.1230e3)))) / density;
  return range;
}
