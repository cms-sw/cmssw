
#ifndef TP_PIXELCHIPINDICES_H
#define TP_PIXELCHIPINDICES_H

#include <iostream>
//#include <vector>
//#include <map>

/**
 * Numbering of the pixels inside the readout chip (ROC).
 * There is a column index and a row index.
 * In the barrel the row index is in the global rfi direction (local X) and
 * the column index is in the global z (local Y) direction.
 * In the endcaps the row index is in the global r direction (local X) and
 * the column index in the gloabl rfi (local Y) direction.
 * The defaults are specific to 100*150 micron pixels.
 * Needs fixing: 
 *   - no limit checking (some added 2/03).
 *   - the 2nd ROC row has inverted indices.
 *   - the 2nd column of each dcol needs reversed pixe counting.
 * 
 * DO NOT USE YET FOR ROC DATA TRANSFORMATION, CAN ONLY BE USED 
 * FOR RATE STUDIES!
 * d.k. 1/2003
 */
using namespace std;
// A few constants just for error checking
// The maximum number of ROCs in the X (row) direction per sensor.
const int maxROCsInX = 2;  //  
// The maximum number of ROCs in the Y (column) direction per sensor.
const int maxROCsInY = 8;  //
// The nominal number of double columns per ROC is 26. Very unlikely
// to ever change, but who knows?
const int nominalDColsPerROC = 26; 
// Default ROC size 
const int defaultROCSizeInX = 80;  // ROC row size in pixels 
const int defaultROCSizeInY = 52;  // ROC col size in pixels 
// Default DET barrel size 
//const int defaultDetSizeInX = 106;  // Det barrel row size in pixels 
const int defaultDetSizeInX = 160;  // Det barrel row size in pixels 
const int defaultDetSizeInY = 416;  // Det barrel col size in pixels 

// Check the limits
#define TP_CHECK_LIMITS

class PixelChipIndices {

 public:

  //*************************************************************************
  // Constructor with all 4 parameters defines explicitly:
  // colsInChip = number of columns in a ROC
  // rowsInChipCol = number of rows in a ROC
  // colsInDet = number of columns in a detector module.
  // rowsInDet = number if rows in a detector module.
  PixelChipIndices(const int colsInChip, const int rowsInChipCol, 
		   const int colsInDet,  const int rowsInDet ) : 
    theColsInChip(colsInChip), theRowsInChipCol(rowsInChipCol), 
    theColsInDet(colsInDet), theRowsInDet (rowsInDet) {

    theChipsInX = theRowsInDet / theRowsInChipCol; // number of ROCs in X
    theChipsInY = theColsInDet / theColsInChip;    // number of ROCs in Y

    // A double column consists of 2 columns
    theDColsInChip = theColsInChip/2;  

#ifdef TP_CHECK_LIMITS
    if(theChipsInX<1 || theChipsInX>maxROCsInX) 
      cout << " PixelChipIndices: Error in ChipsInX " 
	   << theChipsInX << endl;
    if(theChipsInY<1 || theChipsInY>maxROCsInY) 
      cout << " PixelChipIndices: Error in ChipsInY " 
	   << theChipsInY << endl;

    if(theDColsInChip != nominalDColsPerROC ) 
      cout << " PixelChipIndices: Error in DCOLsInChip " 
	   << theDColsInChip << endl;
#endif

    //cout << " Pixel det initlized with " << theChipsInX 
    //<< " chips in x and " << theChipsInY << " in y " << endl; 
  } 
  //*********************************************************************
  // Constructor with the ROC size fixed to default.
  PixelChipIndices(const int colsInDet,  const int rowsInDet ) : 
    theColsInChip(defaultROCSizeInY), theRowsInChipCol(defaultROCSizeInX), 
    theColsInDet(colsInDet), theRowsInDet (rowsInDet) {

    theChipsInX = theRowsInDet / theRowsInChipCol; // number of ROCs in X
    theChipsInY = theColsInDet / theColsInChip;    // number of ROCs in Y

    // A double column consists of 2 columns
    theDColsInChip = theColsInChip/2;  

#ifdef TP_CHECK_LIMITS
    if(theChipsInX<1 || theChipsInX>maxROCsInX) 
      cout << " PixelChipIndices: Error in ChipsInX " 
	   << theChipsInX << endl;
    if(theChipsInY<1 || theChipsInY>maxROCsInY) 
      cout << " PixelChipIndices: Error in ChipsInY " 
	   << theChipsInY << endl;

    if(theDColsInChip != nominalDColsPerROC ) 
      cout << " PixelChipIndices: Error in DCOLsInChip " 
	   << theDColsInChip << endl;
#endif

    //cout << " Pixel det initlized with " << theChipsInX 
    //<< " chips in x and " << theChipsInY << " in y " << endl; 
  } 
  //************************************************************************
  // Constructor with all dimensioned fixed to BARREL default ROC.
  // Usefull only for barrel.
  PixelChipIndices( ) : 
    theColsInChip(defaultROCSizeInY), theRowsInChipCol(defaultROCSizeInX), 
    theColsInDet(defaultDetSizeInY), theRowsInDet (defaultDetSizeInX) {

    theChipsInX = theRowsInDet / theRowsInChipCol;  // number of ROCs in X
    theChipsInY = theColsInDet / theColsInChip;     // number of ROCs in Y

    // A double column consists of 2 columns
    theDColsInChip = theColsInChip/2;  

#ifdef TP_CHECK_LIMITS
    if(theChipsInX<1 || theChipsInX>maxROCsInX) 
      cout << " PixelChipIndices: Error in ChipsInX " 
	   << theChipsInX << endl;
    if(theChipsInY<1 || theChipsInY>maxROCsInY) 
      cout << " PixelChipIndices: Error in ChipsInY " 
	   << theChipsInY << endl;

    if(theDColsInChip != nominalDColsPerROC ) 
      cout << " PixelChipIndices: Error in DCOLsInChip " 
	   << theDColsInChip << endl;
#endif

    //cout << " Pixel det initlized with " << theChipsInX 
    //<< " chips in x and " << theChipsInY << " in y " << endl; 
  } 
  //************************************************************************
  ~PixelChipIndices() {}
  //***********************************************************************
  void print(void) const {
    cout << " Pixel det with " << theChipsInX << " chips in x and "
	 << theChipsInY << " in y " << endl; 
    cout << " Pixel rows " << theRowsInDet << " and columns " 
	 << theColsInDet << endl;  
    cout << " Rows in one chip " << theRowsInChipCol << " and columns " 
	 << theColsInChip << endl;  
    cout << " Double columns per ROC " << theDColsInChip << endl;
  }
  //**********************************************************************
  // Convert detector indices to chip indices. 
  // These are SINGLE COLUMN indices!
  // IN: pixelX - detector index in the X (row) direction, goes from 1 to max
  // IN: pixelY - detector index in the Y (col) direction, goes from 1 to max
  // OUT: chipX - ROC index in the X direction, goes from 0 to 1
  // OUT: chipY - ROC index in the Y direction, goes from 0 to 7 
  // OUT: row - row (X) index within a ROC, goes from 1 to max(80)
  // OUT: col - column (Y) index within a ROC, goes from 1 to max(52) 
  // WARNING: the pixel indices within a ROC are calulated from the WRONG
  //          side. CHECK THIS LATER => Needs to be fixed.
  void chipIndices(const int pixelX, const int pixelY, int & chipX,
		   int & chipY, int & row, int & col) const {

    int pixX = pixelX-1; 
    int pixY = pixelY-1;

    chipX = pixX / theRowsInChipCol; // row index of the chip 0-1
    chipY = pixY / theColsInChip;    // col index of the chip 0-7

    row = (pixX%theRowsInChipCol) + 1; // row in chip
    col = (pixY%theColsInChip) + 1;    // col in chip

#ifdef TP_CHECK_LIMITS
    if(chipX < 0 || chipX > (theChipsInX-1) ) 
      cout << " PixelChipIndices: Error in chipX " << chipX << endl;
    if(chipY < 0 || chipY > (theChipsInY-1) ) 
      cout << " PixelChipIndices: Error in chipY " << chipY << endl;
    if(row < 1 || row > theRowsInChipCol ) 
      cout << " PixelChipIndices: Error in ROC row# " << row << endl;
    if(col < 1 || col > theColsInChip ) 
      cout << " PixelChipIndices: Error in ROC col# " << col << endl;
#endif

  }
  //***********************************************************************
  // Calcilate a single number ROC index from the 2 ROC indices (coordinates)
  // chipX and chipY.
  // Goes from 1 to 16.
  inline static int chipIndex(const int chipX, const int chipY) {
    int chipIndex = (chipX*8) + chipY +1; // index 1-16
#ifdef TP_CHECK_LIMITS
    if(chipIndex < 1 || chipIndex > (maxROCsInX*maxROCsInY) ) 
      cout << " PixelChipIndices: Error in ROC index " << chipIndex << endl;
#endif
    return chipIndex;
  }
  //***********************************************************************
  // Calulate the DCOL indices from single column indices.
  // IN: row = row index witin a single column , goes 1 to max (80)
  // IN: col = single column index, goes 1 to max (52)
  // Returns a pair<pixId, dColumnId>
  // pixId = pixel index within 1 double column, goes from 1 to max (106)
  // dColumnId = double column (DCOL) index, goes from 1 to max (26)
      pair<int,int> dColumn(const int row, const int col) const {
    int dColumnId = (col-1)/2 + 1; // double column 1-26
    int pixId = row + (col-dColumnId*2+1)*theRowsInChipCol; //id in dCol 1-106 
#ifdef TP_CHECK_LIMITS
    if(dColumnId < 1 || dColumnId > theDColsInChip ) 
      cout << " PixelChipIndices: Error in DColumn Id " << dColumnId << endl;
    if(pixId < 1 || pixId > (2*theRowsInChipCol) ) 
      cout << " PixelChipIndices: Error in pix Id " << pixId << endl;
#endif
    return pair<int,int>(pixId,dColumnId);
  }
  //************************************************************************
  // Return a unique double column index with the whole detector module.
  // Usefull only for diagnostics.
  // IN: dcol = DCOL index within a ROC
  // IN: chipOndex = ROC index.
  // Return a 3 digit number XYZ, where X is the ROC index and YZ is 
  // the DCOL index. 
  inline static int dColumnIdInDet(const int dcol, const int chipIndex) {
    int id = dcol + (100*(chipIndex-1));
#ifdef TP_CHECK_LIMITS
    if( id < 1 || 
        id > (((maxROCsInX*maxROCsInY)-1)*100 + nominalDColsPerROC) ) 
      cout << " PixelChipIndices: Error in det dcol id " << id << endl;
#endif

    return id;
  }
  //***********************************************************************
 private:

    int theColsInChip;     // Columns per chip
    int theRowsInChipCol;  // Rows per chip
    int theColsInDet;      // Columns per Det
    int theRowsInDet;      // Rows per Det
    int theChipsInX;       // Chips in det in X (column direction)
    int theChipsInY;       // Chips in det in Y (row direction)
    int theDColsInChip;    // Number of double column per ROC
};

#endif




