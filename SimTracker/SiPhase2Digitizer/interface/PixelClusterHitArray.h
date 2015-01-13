#ifndef SimTracker_SiPhase2Digitizer_HitArray_h
#define SimTracker_SiPhase2Digitizer_HitArray_h

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include <vector>

class PixelClusterHitArray {

public:
        PixelClusterHitArray(int rows, int cols);
	PixelClusterHitArray() { }

	void setSize(int rows, int cols);
	int operator()(int row, int col) const;
	int operator()(const SiPixelCluster::PixelPos&) const;
	int rows() const { return nrows; }
	int columns() const { return ncols; }

	bool inside(int row, int col) const;
	void set(int row, int col, int adc);
	void set(const SiPixelCluster::PixelPos&, int adc);
	int size() const { return pixel_vec.size(); }

	int index(int row, int col) const { return col * nrows + row; }
	int index(const SiPixelCluster::PixelPos& pix) const { return index(pix.row(), pix.col()); }

private:
	int nrows;
	int ncols;
	std::vector<int> pixel_vec;
};

#endif
