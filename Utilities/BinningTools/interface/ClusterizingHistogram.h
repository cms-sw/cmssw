#ifndef _CLUSTERIZINGHISTOGRAM_TT_H_
#define _CLUSTERIZINGHISTOGRAM_TT_H_

#include <vector>
#include <cmath>

/** A very simple 1D equidistant bin histogram that has the ability
 *  to clusterize it's contents.
 *  The bin entries are averaged in X, giving more accurate indication of
 *  where the bin contents are than the center of the bin.
 */

class ClusterizingHistogram {
public:
  ClusterizingHistogram(int nb, float xmi, float xma);
  ~ClusterizingHistogram();

  void fill( float x);
  int nbins() const {return my_nbins;}
  float min_x() const {return xmin;}
  float max_x() const {return xmax;}
  int entries() const {return my_entries;}
  int underflows() const {return my_underflows;}
  int overflows() const {return my_overflows;}
  float bin_pos(int i) const {
    return (bin_entries[i]!=0) ? bin_means[i]/bin_entries[i] : 0;}
  void dump() const;
  void dump(int i1, int i2) const;
  void dump(float x1, float x2) const;
  void dump(double x1, double x2) const;
  void dump(float x1, double x2) const;
  void dump(double x1, float x2) const;
  void reset();
  int bin( float x) const;
  int bin( double x) const;

  std::vector<float> clusterize( float resolution);
  
private:
  ClusterizingHistogram(){}  // Prohibit
  int my_nbins;
  float xmin;
  float xmax;
  int my_entries;
  int my_underflows;
  int my_overflows;
  int *bin_entries;
  float *bin_means;
  float binsiz;
};

#endif
