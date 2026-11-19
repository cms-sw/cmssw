#pragma once

// system include files
#include <vector>

// user include files
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

/// Cartesian seeding grid, intended for cosmic track reconstruction.
/// Collects space-points in an x-y-z binning in the global frame.
/// In each bin, orders the hits by descending y.
/// Does not take ownership of the hits - user is responsible
/// for ensuring pointer validity and object lifetime.
class CartesianSeedingGrid {
public:
  /// @brief Constructor, taking the parameters of the (rectangular) bin grid.
  /// @param nBinsX: Number of bins in the global x coordinate
  /// @param xmin: Low edge of the first bin in global x
  /// @param xmax: Up edge of the last bin in global x
  /// @param nBinsY: Number of bins in the global y coordinate
  /// @param ymin: Low edge of the first bin in global y
  /// @param ymax: Up edge of the last bin in global y
  /// @param nBinsZ: Number of bins in the global z coordinate
  /// @param zmin: Low edge of the first bin in global z
  /// @param zmax: Up edge of the last bin in global z
  CartesianSeedingGrid(
      int nBinsX, double xmin, double xmax, int nBinsY, double ymin, double ymax, int nBinsZ, double zmin, double zmax);

  /// @brief add a hit to the grid. Will be placed in a bin corresponding
  /// to the coordinates returned by its globalPosition().
  /// @param h: Hit to be stored.
  void addHit(const BaseTrackerRecHit* h);

  /// @brief convert a global x-coordinate to the corresponding bin index.
  /// Under/overflows will be added to the first/last bins.
  /// @param x global x coordinate value
  /// @return bin index in the x coordinate.
  inline int binX(double x) const { return findBin(x, nBinsX_, xmin_, xmax_); }

  /// @brief convert a global y-coordinate to the corresponding bin index.
  /// Under/overflows will be added to the first/last bins.
  /// @param y global y coordinate value
  /// @return bin index in the y coordinate.
  inline int binY(double y) const { return findBin(y, nBinsY_, ymin_, ymax_); }
  /// @brief convert a global z-coordinate to the corresponding bin index.
  /// Under/overflows will be added to the first/last bins.
  /// @param z global y coordinate value
  /// @return bin index in the z coordinate.
  inline int binZ(double z) const { return findBin(z, nBinsZ_, zmin_, zmax_); }

  /// @brief access (read-only) the hit content of one cell.
  /// @param binX: The x-bin-index
  /// @param binY: The y-bin-index
  /// @param binZ: The z-bin-index
  /// @return the hits stored in cell (binX,binY,binZ), as a read-only reference
  const std::vector<const BaseTrackerRecHit*>& getHits(int binX, int binY, int binZ) const;

  /// @brief sort the hits in each cell to be ordered descending in global y.
  /// Call this **after** filling the binning grid with all hits to be considered.
  void sort();

  /// @brief helper function - maps 3D bin indices to our 1D vector.
  /// @param bx: index of the bin along x
  /// @param by: index of the bin along y
  /// @param bz: index of the bin along z
  /// @return Index of the corresponding cell in our storage vector.
  int getBin(int bx, int by, int bz) const;

  /// @brief getter for the number of bins along x.
  /// @return the number of bins along x
  inline int nBinsX() const { return nBinsX_; }

  /// @brief getter for the number of bins along y.
  /// @return the number of bins along y
  inline int nBinsY() const { return nBinsY_; }

  /// @brief getter for the number of bins along z.
  /// @return the number of bins along z
  inline int nBinsZ() const { return nBinsZ_; }

  /// @brief getter for the starting coordinate of the x-binning
  /// @return the low edge of the first bin along x
  inline double xmin() const { return xmin_; }
  /// @brief getter for the end coordinate of the x-binning
  /// @return the up edge of the last bin along x
  inline double xmax() const { return xmax_; }
  /// @brief getter for the starting coordinate of the y-binning
  /// @return the low edge of the first bin along y
  inline double ymin() const { return ymin_; }
  /// @brief getter for the end coordinate of the y-binning
  /// @return the up edge of the last bin along y
  inline double ymax() const { return ymax_; }
  /// @brief getter for the starting coordinate of the z-binning
  /// @return the low edge of the first bin along z
  inline double zmin() const { return zmin_; }
  /// @brief getter for the end coordinate of the z-binning
  /// @return the up edge of the last bin along z
  inline double zmax() const { return zmax_; }

private:
  /// @brief helper function to locate a bin. Will return the index of the bin in a
  /// (equidistant) binning of the interval [min,max] into nBins bins.
  /// Bins are indexed in "C-convention", meaning from 0 to nBins - 1.
  /// @param value: Value to translate to a bin index
  /// @param nBins: Number of bins
  /// @param min: Low edge of the binned interval
  /// @param max: Up edge of the binned interval
  /// @return Integer in the range [0, nBins-1] indicating the bin in which
  /// the passed value lies.
  /// Under/Overflow will be merged into the first / last bin.
  int findBin(double value, int nBins, double min, double max) const;

  /// data members

  // x binning
  int nBinsX_;   /// number of bins on x axis
  double xmin_;  /// low edge of x axis
  double xmax_;  /// up edge of x axis

  // y binning
  int nBinsY_;   /// number of bins on y axis
  double ymin_;  /// low edge of y axis
  double ymax_;  /// up edge of y axis

  // z binning
  int nBinsZ_;   /// number of bins on z axis
  double zmin_;  /// low edge of z axis
  double zmax_;  /// up edge of z axis

  /// Data storage vector. Contains a vector of hits for each cell.
  /// Outer vector has size nBinsX_ * nBinsY_ * nBinsZ_, inner vector for
  /// each cell grows dynamically.
  std::vector<std::vector<const BaseTrackerRecHit*>> recHits_ = {};
};
