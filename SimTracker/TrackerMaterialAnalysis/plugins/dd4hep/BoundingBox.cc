#include "BoundingBox.h"

void BoundingBox::grow(double r, double z) {
  if (r < r_min)
    r_min = r;
  if (r > r_max)
    r_max = r;
  if (z < z_min)
    z_min = z;
  if (z > z_max)
    z_max = z;
}

void BoundingBox::grow(double skin) {
  r_min -= skin;  // yes, we allow r_min to go negative
  r_max += skin;
  z_min -= skin;
  z_max += skin;
}

bool BoundingBox::inside(double r, double z) const { return (r >= r_min and r <= r_max and z >= z_min and z <= z_max); }
