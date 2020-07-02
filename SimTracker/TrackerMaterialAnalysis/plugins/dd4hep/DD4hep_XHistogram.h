#ifndef DD4hep_XHistogram_h
#define DD4hep_XHistogram_h

#include <algorithm>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>

#include <TH2F.h>
#include <TH2I.h>

class DD4hep_XHistogram {
public:
  typedef TH2I ColorMap;
  typedef TH2F Histogram;
  typedef std::pair<double, double> Range;

protected:
  double m_minDl;
  Range m_xRange;
  Range m_yRange;
  size_t m_xBins;
  size_t m_yBins;
  size_t m_size;

  std::vector<std::shared_ptr<Histogram> > m_histograms;
  std::shared_ptr<Histogram> m_normalization;
  std::shared_ptr<ColorMap> m_colormap;
  std::shared_ptr<Histogram> m_dummy;

public:
  /// default CTOR
  DD4hep_XHistogram(void)
      : m_minDl(0.000001),
        m_xRange(),
        m_yRange(),
        m_xBins(),
        m_yBins(),
        m_size(),
        m_histograms(),
        m_normalization(),
        m_colormap(),
        m_dummy() {}

  // explicit CTOR
  DD4hep_XHistogram(
      size_t size, size_t bins_x, size_t bins_y, Range x, Range y, size_t zones, const std::vector<double>& max)
      : m_minDl(0.000001),
        m_xRange(x),
        m_yRange(y),
        m_xBins(bins_x),
        m_yBins(bins_y),
        m_size(size),
        m_histograms(m_size),
        m_normalization(),
        m_colormap() {
    // setup unnamed ROOT histograms
    for (size_t i = 0; i < m_size; ++i) {
      m_histograms[i].reset(new Histogram(nullptr, nullptr, bins_x, x.first, x.second, bins_y, y.first, y.second));
      m_histograms[i]->SetMinimum(0.);
      m_histograms[i]->SetMaximum(max[i]);
    }
    m_normalization.reset(new Histogram(nullptr, nullptr, bins_x, x.first, x.second, bins_y, y.first, y.second));
    m_colormap.reset(new ColorMap(nullptr, nullptr, bins_x, x.first, x.second, bins_y, y.first, y.second));
    m_colormap->SetMinimum(0);
    m_colormap->SetMaximum(zones);
    Histogram(nullptr, nullptr, 0, 0., 0., 0, 0., 0.);  // make ROOT "forget" about unnamed histograms
  }

  /// fill one point
  void fill(double x, double y, const std::vector<double>& weight, double norm);

  /// fill one point and set its color
  void fill(double x, double y, const std::vector<double>& weight, double norm, unsigned int colour);

  /// fill one segment, normalizing each bin's weight to the fraction of the segment it contains
  void fill(const Range& x, const Range& y, const std::vector<double>& weight, double norm);

  /// fill one segment and set its color, normalizing each bin's weight to the fraction of the segment it contains
  void fill(const Range& x, const Range& y, const std::vector<double>& weight, double norm, unsigned int colour);

  /// normalize the histograms
  void normalize(void);

  /// access one of the histograms
  Histogram* get(size_t h = 0) const {
    if (h < m_size)
      return (Histogram*)m_histograms[h]->Clone(nullptr);
    else
      return nullptr;
  }

  /// access the normalization
  Histogram* normalization(void) const { return (Histogram*)m_normalization->Clone(nullptr); }

  /// access the colormap
  ColorMap* colormap(void) const { return (ColorMap*)m_colormap->Clone(nullptr); }

  /// set the minimum length of sub-segment a segment should be split into:
  /// when splitting across bin boundaries with splitSegment(...), sub-segments shorter than this are skipped
  void setMinDl(double dl) { m_minDl = dl; }

protected:
  struct position {
    double f;
    double x;
    double y;

    position() : f(0), x(0), y(0) {}

    position(double f_, double x_, double y_) : f(f_), x(x_), y(y_) {}

    bool operator<(const position& other) const { return f < other.f; }
  };

  /// split a segment into a vector of points
  std::vector<position> splitSegment(Range x, Range y) const;

  /// check the weights passed as an std::vector have the correct size
  void check_weight(const std::vector<double>& weight) noexcept(false) {
    // run time check for vector size
    if (weight.size() != m_size)
      throw std::invalid_argument("weight: wrong number of elements");
  }
};

#endif  // DD4hep_XHistogram_h
