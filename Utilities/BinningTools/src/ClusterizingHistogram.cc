#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include <iostream>

using namespace std;

ClusterizingHistogram::ClusterizingHistogram(int nb, float xmi, float xma) : 
  my_nbins(nb), xmin(xmi), xmax(xma), my_entries(0), my_underflows(0), my_overflows(0) {
    bin_entries = new int[my_nbins];
    bin_means = new float[my_nbins];
    binsiz = (xmax-xmin) / my_nbins;
    for (int i=0; i<my_nbins; i++) {
      bin_entries[i] = 0;
      bin_means[i] = 0.;
    }
}
ClusterizingHistogram::~ClusterizingHistogram(){ delete[] bin_entries; delete [] bin_means;}

int ClusterizingHistogram::bin( float x) const {
  if      (x < xmin) return -1;
  else if (x > xmax) return my_nbins;
  else return int((x-xmin)/binsiz);
}
int ClusterizingHistogram::bin( double x) const {
  if      (x < xmin) return -1;
  else if (x > xmax) return my_nbins;
  else return int((x-xmin)/binsiz);
}

void ClusterizingHistogram::fill( float x) {
  if      (x < xmin) my_underflows++;
  else if (x > xmax) my_overflows++;
  else {
    int bin = int((x-xmin)/binsiz);
    if ( bin > my_nbins-1) bin = my_nbins-1;
    ++bin_entries[bin];
    bin_means[bin] += x;
    my_entries++;
    // may be problematic for negative x; check!
  }
}

vector<float> ClusterizingHistogram::clusterize( float resol) {
  vector<float> clust;
  int nclust = 0;
  bool inclust = false;
  float last_pos = xmin - 1000.*resol;
  int sum = 0;
  float sumx = 0;
  for (int i=0; i<my_nbins; i++) {
    if (bin_entries[i] != 0) {
      if ( fabs(bin_pos(i)-last_pos) > resol) {
	inclust = false;
	if (nclust != 0) clust.push_back( sumx/sum);  // create cluster
      }
      if (!inclust) {
	nclust++;
        sumx = 0.;
	sum = 0;
      }
      sum += bin_entries[i];
      sumx += bin_means[i];
      last_pos = bin_pos(i);
      inclust = true;
    }
  }
  if (nclust != 0) clust.push_back( sumx/sum);  // create last cluster
  return clust;
}


void ClusterizingHistogram::dump() const { dump(0, my_nbins);}

void ClusterizingHistogram::dump(int i1, int i2) const {
  cout << "Dumping ClusterizingHistogram contents:" << endl;
  for (int i=max(i1,0); i<min(i2,my_nbins); i++) {
    cout << i << "  " << bin_entries[i] << "   " << bin_pos(i) << endl;
  }
  cout << "Underflows: " << my_underflows << endl;
  cout << "Overflows:  " << my_overflows << endl;
  cout << "Total number of entries: " << my_entries << endl;
}

void ClusterizingHistogram::dump( float x1, float x2) const { dump( bin(x1), bin(x2));}
void ClusterizingHistogram::dump( double x1, double x2) const { dump( bin(x1), bin(x2));}
void ClusterizingHistogram::dump( float x1, double x2) const { dump( bin(x1), bin(x2));}
void ClusterizingHistogram::dump( double x1, float x2) const { dump( bin(x1), bin(x2));}

void ClusterizingHistogram::reset() {
  my_entries = 0;
  my_underflows = 0;
  my_overflows = 0;
  for (int i=0; i<my_nbins; i++) {
    bin_entries[i] = 0;
    bin_means[i] = 0.;
  }
}
