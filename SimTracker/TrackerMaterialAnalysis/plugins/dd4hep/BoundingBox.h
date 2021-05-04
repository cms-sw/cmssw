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

  BoundingBox(const double& min_r, const double& max_r, const double& min_z, const double& max_z)
      : r_min(min_r), r_max(max_r), z_min(min_z), z_max(max_z) {}

  void grow(const double& r, const double& z);

  void grow(const double& skin);

  inline bool inside(const double& r, const double& z) const {
    return (r >= r_min and r <= r_max and z >= z_min and z <= z_max);
  };

  std::pair<double, double> range_r() const { return std::make_pair(r_min, r_max); }

  std::pair<double, double> range_z() const { return std::make_pair(z_min, z_max); }
};

#endif
