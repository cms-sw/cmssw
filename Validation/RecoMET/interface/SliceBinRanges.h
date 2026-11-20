#ifndef Validation_RecoMET_SliceBinRanges_h
#define Validation_RecoMET_SliceBinRanges_h

#include "TAxis.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace metvalidation {

  // Slice edges are floats and axis edges are doubles computed from float constants,
  // so they can differ by rounding errors: a tolerance, as a fraction of the bin width, is therefore needed.
  inline int snapToEdge(const TAxis& ax, double x) {
    constexpr double tolerance = 1e-3;
    const int bin = std::clamp(ax.FindFixBin(x), 1, ax.GetNbins());
    const double margin = tolerance * ax.GetBinWidth(bin);
    if (std::abs(x - ax.GetBinLowEdge(bin)) <= margin)
      return bin - 1;
    if (std::abs(x - ax.GetBinUpEdge(bin)) <= margin)
      return bin;
    return -1;
  }

  inline std::vector<std::pair<int, int>> sliceBinRanges(const TAxis& ax, const std::vector<float>& bins) {
    std::vector<std::pair<int, int>> ranges;
    for (unsigned binId = 0; binId + 1 < bins.size(); binId++) {
      // the upper edge belongs to the next slice
      const int lowEdge = snapToEdge(ax, bins[binId]);
      const int upEdge = snapToEdge(ax, bins[binId + 1]);
      const int first = lowEdge >= 0 ? lowEdge + 1 : ax.FindFixBin(bins[binId]);
      // ProjectionX and ProjectionY include both 'first' and 'last' edges in the projection
      int last = upEdge >= 0 ? upEdge : ax.FindFixBin(bins[binId + 1]);
      // ROOT projects the whole axis if last < first
      last = std::max(first, last);
      ranges.emplace_back(first, last);
    }
    return ranges;
  }

  // Indices (in `bins`) of the interior slice edges that do not fall on a bin edge of the axis: the bin containing
  // such an edge is shared by the two neighbouring slices. The outer edges may lie inside a bin without consequences.
  inline std::vector<unsigned> misalignedSliceEdges(const TAxis& ax, const std::vector<float>& bins) {
    std::vector<unsigned> misaligned;
    for (unsigned i = 1; i < bins.size() - 1; ++i)
      if (snapToEdge(ax, bins[i]) < 0)
        misaligned.push_back(i);
    return misaligned;
  }

}  // namespace metvalidation

#endif  // Validation_RecoMET_SliceBinRanges_h
