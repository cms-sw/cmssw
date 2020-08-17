#include "BoundingBox.h"

void BoundingBox::grow(const double& r, const double& z) {
  if (r < r_min)
    r_min = r;
  if (r > r_max)
    r_max = r;
  if (z < z_min)
    z_min = z;
  if (z > z_max)
    z_max = z;
}

void BoundingBox::grow(const double& skin) {
  r_min -= skin;  // yes, we allow r_min to go negative
  r_max += skin;
  z_min -= skin;
  z_max += skin;
}
