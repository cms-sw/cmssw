#include <algorithm>
#include <cmath>
#include <vector>

#include "XHistogram.h"

std::vector<XHistogram::position> XHistogram::splitSegment(Range rangeX, Range rangeY) const {
  double deltaX = rangeX.second - rangeX.first;
  double deltaY = rangeY.second - rangeY.first;
  double length = hypot(deltaX, deltaY);
  double stepX = (m_xRange.second - m_xRange.first) / m_xBins;
  double stepY = (m_yRange.second - m_yRange.first) / m_yBins;

  int min_i, max_i, min_j, max_j;
  if (rangeX.first < rangeX.second) {
    min_i = (int)ceil(rangeX.first / stepX);        // included
    max_i = (int)floor(rangeX.second / stepX) + 1;  // excluded
  } else {
    min_i = (int)ceil(rangeX.second / stepX);
    max_i = (int)floor(rangeX.first / stepX) + 1;
  }
  if (rangeY.first < rangeY.second) {
    min_j = (int)ceil(rangeY.first / stepY);
    max_j = (int)floor(rangeY.second / stepY) + 1;
  } else {
    min_j = (int)ceil(rangeY.second / stepY);
    max_j = (int)floor(rangeY.first / stepY) + 1;
  }

  int steps = max_i - min_i + max_j - min_j + 2;
  std::vector<position> v;
  v.clear();
  v.reserve(steps);

  v.push_back(position(0., rangeX.first, rangeY.first));
  double x, y, f;
  for (int i = min_i; i < max_i; ++i) {
    x = i * stepX;
    y = rangeY.first + (x - rangeX.first) * deltaY / deltaX;
    f = std::fabs((x - rangeX.first) / deltaX);
    v.push_back(position(f, x, y));
  }
  for (int i = min_j; i < max_j; ++i) {
    y = i * stepY;
    x = rangeX.first + (y - rangeY.first) * deltaX / deltaY;
    f = std::fabs((y - rangeY.first) / deltaY);
    v.push_back(position(f, x, y));
  }
  v.push_back(position(1., rangeX.second, rangeY.second));

  // sort by distance from the start of the segment
  std::sort(v.begin(), v.end());

  // filter away the fragments shorter than m_minDl, and save the center of each fragment along with its fractionary length
  std::vector<position> result;
  result.push_back(v.front());
  for (int i = 1, s = v.size(); i < s; ++i) {
    double mx = (v[i].x + v[i - 1].x) / 2.;
    double my = (v[i].y + v[i - 1].y) / 2.;
    double df = (v[i].f - v[i - 1].f);
    if (df * length < m_minDl)
      continue;
    result.push_back(position(df, mx, my));
  }

  return result;
}

/// fill one point
void XHistogram::fill(double x, double y, const std::vector<double>& weight, double norm) {
  check_weight(weight);

  for (size_t h = 0; h < m_size; ++h)
    m_histograms[h]->Fill(x, y, weight[h]);
  m_normalization->Fill(x, y, norm);
}

/// fill one point and set its color
void XHistogram::fill(double x, double y, const std::vector<double>& weight, double norm, unsigned int colour) {
  check_weight(weight);

  for (size_t h = 0; h < m_size; ++h)
    m_histograms[h]->Fill(x, y, weight[h]);
  m_normalization->Fill(x, y, norm);
  m_colormap->SetBinContent(m_colormap->FindBin(x, y), (float)colour);
}

/// fill one segment, normalizing each bin's weight to the fraction of the segment it contains
void XHistogram::fill(const Range& x, const Range& y, const std::vector<double>& weight, double norm) {
  check_weight(weight);

  std::vector<position> v = splitSegment(x, y);
  for (size_t i = 0, s = v.size(); i < s; ++i) {
    for (size_t h = 0; h < m_size; ++h)
      m_histograms[h]->Fill(v[i].x, v[i].y, v[i].f * weight[h]);
    m_normalization->Fill(v[i].x, v[i].y, v[i].f * norm);
  }
}

/// fill one segment and set its color, normalizing each bin's weight to the fraction of the segment it contains
void XHistogram::fill(
    const Range& x, const Range& y, const std::vector<double>& weight, double norm, unsigned int colour) {
  check_weight(weight);

  std::vector<position> v = splitSegment(x, y);
  for (size_t i = 0, s = v.size(); i < s; ++i) {
    for (size_t h = 0; h < m_size; ++h)
      m_histograms[h]->Fill(v[i].x, v[i].y, v[i].f * weight[h]);
    m_normalization->Fill(v[i].x, v[i].y, v[i].f * norm);
    m_colormap->SetBinContent(m_colormap->FindBin(v[i].x, v[i].y), (float)colour);
  }
}

/// normalize the histograms
void XHistogram::normalize(void) {
  for (int i = 0; i < m_normalization->GetSize(); ++i) {
    if ((*m_normalization)[i] > 0.) {
      for (size_t h = 0; h < m_size; ++h)
        (*m_histograms[h])[i] /= (*m_normalization)[i];
      (*m_normalization)[i] = 1.;
    }
  }
}
