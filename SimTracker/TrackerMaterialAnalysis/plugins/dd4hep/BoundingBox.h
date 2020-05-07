#ifndef BoundingBox_h
#define BoundingBox_h

#include <utility>

class BoundingBox {
private:
  double r_min;
  double r_max;
  double z_min;
  double z_max;

public:
  BoundingBox() : r_min(0.), r_max(0.), z_min(0.), z_max(0.) {}

  BoundingBox(double min_r, double max_r, double min_z, double max_z)
      : r_min(min_r), r_max(max_r), z_min(min_z), z_max(max_z) {}

  void grow(double r, double z);

  void grow(double skin);

  bool inside(double r, double z) const;

  std::pair<double, double> range_r() const { return std::make_pair(r_min, r_max); }

  std::pair<double, double> range_z() const { return std::make_pair(z_min, z_max); }
};

#endif
