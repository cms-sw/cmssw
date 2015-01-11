#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterHitArray.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

// Create a 2D matrix of zeros of size rows x cols
PixelClusterHitArray::PixelClusterHitArray(int rows, int cols) :
    nrows(rows),
    ncols(cols) {
    pixel_vec.resize(rows * cols);
    for (std::vector<int>::iterator it = pixel_vec.begin(); it != pixel_vec.end(); ++it) *it = 0;
}

// Change the size of the matrix and reset it
void PixelClusterHitArray::setSize(int rows, int cols) {
    nrows = rows;
    ncols = cols;
    pixel_vec.resize(rows * cols);
    for (std::vector<int>::iterator it = pixel_vec.begin(); it != pixel_vec.end(); ++it) *it = 0;
}

// Check if an element is inside the matrix (no overflow)
bool PixelClusterHitArray::inside(int row, int col) const {
    return (row >= 0 && row < nrows && col >= 0 && col < ncols);
}

// Return an element of the matrix
int PixelClusterHitArray::operator()(int row, int col) const {
    if (inside(row,col)) return pixel_vec[index(row,col)];
    else return 0;
}

// Return an element of the matrix (get the row and column from the pixel data)
int PixelClusterHitArray::operator()(const SiPixelCluster::PixelPos& pix) const {
    if (inside(pix.row(), pix.col())) return pixel_vec[index(pix)];
    else return 0;
}

// Set an element of the matrix
void PixelClusterHitArray::set(int row, int col, int adc) {
    pixel_vec[index(row,col)] = adc;
}

// Set an element of the matrix (get the row and column from the pixel data)
void PixelClusterHitArray::set(const SiPixelCluster::PixelPos& pix, int adc) {
    pixel_vec[index(pix)] = adc;
}
