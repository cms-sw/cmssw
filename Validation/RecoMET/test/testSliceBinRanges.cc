// Checks that sliceBinRanges() assigns every bin of an axis to at most one slice, and to the right one.
// The axes and slices reproduce the ones booked by METTester and sliced by METTesterPostProcessor.

#include "Validation/RecoMET/interface/SliceBinRanges.h"

#include "TAxis.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

  struct Case {
    std::string name;
    int nbins;
    double low, high;  // as passed to TH1 by METTester: float constants converted to double
    std::vector<float> slices;
  };

  // returns the number of problems found
  int check(const Case& c) {
    const TAxis ax(c.nbins, c.low, c.high);
    const auto ranges = metvalidation::sliceBinRanges(ax, c.slices);
    int problems = 0;
    auto report = [&](const std::string& msg) {
      std::cout << "  [" << c.name << "] " << msg << "\n";
      ++problems;
    };

    if (ranges.size() + 1 != c.slices.size())
      report("wrong number of slices");

    for (const auto i : metvalidation::misalignedSliceEdges(ax, c.slices))
      report("slice edge " + std::to_string(c.slices[i]) + " reported as not being on a bin edge");

    for (int bin = 1; bin <= c.nbins; ++bin) {
      // expected slice: the one containing the bin center (unambiguous for slice edges on bin edges)
      const double center = ax.GetBinCenter(bin);
      int expected = -1;
      for (unsigned i = 0; i + 1 < c.slices.size(); ++i)
        if (center >= c.slices[i] and center < c.slices[i + 1])
          expected = i;

      std::vector<unsigned> assigned;
      for (unsigned i = 0; i < ranges.size(); ++i)
        if (bin >= ranges[i].first and bin <= ranges[i].second)
          assigned.push_back(i);

      if (assigned.size() > 1) {
        std::string slices;
        for (auto i : assigned)
          slices += " " + std::to_string(i);
        report("bin " + std::to_string(bin) + " [" + std::to_string(ax.GetBinLowEdge(bin)) + ", " +
               std::to_string(ax.GetBinUpEdge(bin)) + ") assigned to " + std::to_string(assigned.size()) +
               " slices:" + slices);
      } else if (expected >= 0 and (assigned.size() != 1 or static_cast<int>(assigned[0]) != expected)) {
        report("bin " + std::to_string(bin) + " should be in slice " + std::to_string(expected) + " but is in " +
               (assigned.empty() ? std::string("none") : "slice " + std::to_string(assigned[0])));
      } else if (expected < 0 and not assigned.empty()) {
        report("bin " + std::to_string(bin) + " is outside all slices but assigned to slice " +
               std::to_string(assigned[0]));
      }
    }
    return problems;
  }

  // slice edges inside a bin cannot be supported (that bin is shared by two slices): they must be detected
  int checkMisaligned(const Case& c, const std::vector<unsigned>& expected) {
    const TAxis ax(c.nbins, c.low, c.high);
    const auto found = metvalidation::misalignedSliceEdges(ax, c.slices);
    if (found == expected)
      return 0;
    std::string msg;
    for (const auto i : found)
      msg += " " + std::to_string(i);
    std::cout << "  [" << c.name << "] misaligned edges found (indices):" << (msg.empty() ? " none" : msg)
              << ", expected " << expected.size() << " of them\n";
    return 1;
  }

}  // namespace

int main() {
  // constants as in METTester.h (float), converted to double when booking the histograms
  const float phiEdge = 3.2;
  const float metEdge = 1100.;
  const float nvtxEdge = 350.;

  // slices as in METTesterPostProcessor.h
  const std::vector<float> phiBins = {-3.15, -2., -1., 0., 1., 2., 3.15};
  const std::vector<float> metBins = {0., 20., 40., 60., 80., 100., 150., 200., 300., 400., 500., 1000.};
  std::vector<float> nvtxBins;
  for (int i = 0; i <= 35; ++i)
    nvtxBins.push_back(10.f * i);

  const std::vector<Case> cases = {
      {"phi axis (METTester, float edge)", 32, -phiEdge, phiEdge, phiBins},
      {"MET axis", 110, 0., metEdge, metBins},
      {"nvtx axis", 35, 0., nvtxEdge, nvtxBins},
      // same phi axis with the exact double 3.2, to tell whether the problem comes from the float edge
      {"phi axis (double edge)", 32, -3.2, 3.2, phiBins},
  };

  int problems = 0;
  // mid-bin interior edges (0.1 and 1.9 on the phi axis, whose bins are 0.2 wide) are detected, outer ones are not
  // all tests have a bin width of 0.2 radians, and the outer edges (first and last), even if mis-aligned,
  // do not count for the test since they cannot lead to double counting
  const std::vector<std::pair<Case, std::vector<unsigned>>> misaligned = {
      {{"edge inside a bin (0.1)", 32, -phiEdge, phiEdge, {-3.15, 0., 0.1, 1., 3.15}}, {2}},
      {{"edge inside a bin (1.9)", 32, -phiEdge, phiEdge, {0., 1.9, 3.15}}, {1}},
      {{"two edges inside bins", 32, -phiEdge, phiEdge, {-3.15, -2.5, 0.1, 3.15}}, {1, 2}},
      {{"only outer edges inside bins", 32, -phiEdge, phiEdge, {-3.15, 0., 3.15}}, {}},
  };
  for (const auto& [c, expected] : misaligned) {
    const int n = checkMisaligned(c, expected);
    std::cout << (n ? "FAIL " : "ok   ") << c.name << "\n";
    problems += n;
  }

  for (const auto& c : cases) {
    const int n = check(c);
    std::cout << (n ? "FAIL " : "ok   ") << c.name << (n ? ": " + std::to_string(n) + " problem(s)" : "") << "\n";
    problems += n;
  }
  return problems ? 1 : 0;
}
