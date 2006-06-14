#ifndef TP_PIXELINDICES_H
#define TP_PIXELINDICES_H

#include <iostream>
using namespace std;

/**
 * Numbering of the pixels inside the readout chip (ROC).
 * There is a column index and a row index.
 * In the barrel the row index is in the global rfi direction (local X) and
 * the column index is in the global z (local Y) direction.
 * In the endcaps the row index is in the global r direction (local X) and
 * the column index in the gloabl rfi (local Y) direction.
 * The defaults are specific to 100*150 micron pixels.
 * 
 * Some methods are declared as static and can be used without class 
 * instantiation. Others need the construstor to setup the Module size
 * parameters. These parameters are only used for error checking.
 * d.k. 10/2005
 */

namespace {
  // A few constants just for error checking
  // The maximum number of ROCs in the X (row) direction per sensor.
  const int maxROCsInX = 2;  //  
  // The maximum number of ROCs in the Y (column) direction per sensor.
  const int maxROCsInY = 8;  //
  // The nominal number of double columns per ROC is 26. 
  const int DColsPerROC = 26; 
  // Default ROC size 
  const int ROCSizeInX = 80;  // ROC row size in pixels 
  const int ROCSizeInY = 52;  // ROC col size in pixels 
  // Default DET barrel size 
  const int defaultDetSizeInX = 160;  // Det barrel row size in pixels 
  const int defaultDetSizeInY = 416;  // Det barrel col size in pixels 
  
  // Check the limits
  const bool TP_CHECK_LIMITS = true;
}

class PixelIndices {

 public:
  
  //*********************************************************************
  // Constructor with the ROC size fixed to the default.
   PixelIndices(const int colsInDet,  const int rowsInDet ) : 
                theColsInDet(colsInDet), theRowsInDet (rowsInDet) {

    theChipsInX = theRowsInDet / ROCSizeInX; // number of ROCs in X
    theChipsInY = theColsInDet / ROCSizeInY;    // number of ROCs in Y

    if(TP_CHECK_LIMITS) {
      if(theChipsInX<1 || theChipsInX>maxROCsInX) 
	cout << " PixelIndices: Error in ROCsInX " 
	     << theChipsInX <<" "<<theRowsInDet<<" "<<ROCSizeInX<<endl;
      if(theChipsInY<1 || theChipsInY>maxROCsInY) 
	cout << " PixelIndices: Error in ROCsInY " 
	     << theChipsInY <<" "<<theColsInDet<<" "<<ROCSizeInY<<endl;
    }
  } 
  //************************************************************************
  ~PixelIndices() {}
  //***********************************************************************
  void print(void) const {
    cout << " Pixel det with " << theChipsInX << " chips in x and "
	 << theChipsInY << " in y " << endl; 
    cout << " Pixel rows " << theRowsInDet << " and columns " 
	 << theColsInDet << endl;  
    cout << " Rows in one chip " << ROCSizeInX << " and columns " 
	 << ROCSizeInY << endl;  
    cout << " Double columns per ROC " << DColsPerROC << endl;
  }

  //********************************************************************
  // Convert dcol & pix indices to ROC col and row
  // Decoding from "Weber" pixel addresses to rows for PSI46
  // dcol = 0 - 25
  // pix = 2 - 161, zigzag pattern.
  // colAdd = 0-51   ! col&row start from 0
  // rowAdd = 0-79
  inline static int convertDcolToCol(const int dcol, const int pix, 
				     int & colROC, int & rowROC) {
      if(TP_CHECK_LIMITS) { 
	if(dcol<0||dcol>=DColsPerROC||pix<2||pix>161) {
	  cout<<"PixelIndices: wrong dcol or pix "<<dcol<<" "<<pix<<endl;
	  rowROC = -1;     // dummy row Address
	  colROC = -1;     // dummy col Address
	  return -1; // Signal error
	}
      }

      // First find if we are in the first or 2nd col of a dcol.
      int colEvenOdd = pix%2;  // module(2), 0-1st sol, 1-2nd col.
      // Transform
      colROC = dcol * 2 + colEvenOdd; // col address, starts from 0
      rowROC = abs( int(pix/2) - 80); // row addres, starts from 0

      if(TP_CHECK_LIMITS) {
	if(colROC<0||colROC>=ROCSizeInY||rowROC<0||rowROC>=ROCSizeInX ) {
	  cout<<"PixelIndices: wrong col or row "<<colROC<<" "<<rowROC<<" "
	      <<dcol<<" "<<pix<<endl;
	  rowROC = -1;    // dummy row Address
	  colROC = -1;    // dummy col Address
	  return -1;
	}
      }
      return 0;
    }

 //********************************************************************
 // colROC, rowROC are coordinates in the ROC frame, for ROC=rocId
 // (Start from 0).
 // cols, row are coordinates in the module frame, start from 0.
 // row is X, col is Y.
 // At the moment this works only for modules read with a single TBM.
  int transformToModule(const int colROC,const int rowROC,
			const int rocId,
			int & col,int & row ) const {

      if(TP_CHECK_LIMITS) {
	if(colROC<0 || colROC>=ROCSizeInY || rowROC<0 ||rowROC>=ROCSizeInX) {
	  cout<<"PixelIndices: wrong index "<<colROC<<" "<<rowROC<<endl;
	  return -1;
	}
      }

      // The transformation depends on the ROC-ID
      if(rocId>=0 && rocId<8) {
	row = 159-rowROC;
	//col = rocId*52 + colROC;
	col = (8-rocId)*ROCSizeInY - colROC - 1;
      } else if(rocId>=8 && rocId<16) {
	row = rowROC;
	//col = (16-rocId)*52 - colROC - 1;
	col = (rocId-8)*ROCSizeInY + colROC;
      } else {
	cout<<"PixelIndices: wrong ROC ID "<<rocId<<endl;
	return -1;
      }
      if(TP_CHECK_LIMITS) {
	if(col<0 || col>=(ROCSizeInY*theChipsInY) || row<0 || 
			     row>=(ROCSizeInX*theChipsInX)) {
	cout<<"PixelIndices: wrong index "<<col<<" "<<row<<endl;
	return -1;
	}
      }

      return 0;
  }
  //**************************************************************************
  // Transform from the module indixes to the ROC indices.
  // col, row - indices in the Module
  // rocId - roc index
  // colROC, rowROC - indices in the ROC frame.
  int transformToROC(const int col,const int row,
		     int & rocId, int & colROC, int & rowROC ) const {
      if(TP_CHECK_LIMITS) {
	if(col<0 || col>=(ROCSizeInY*theChipsInY) || row<0 || 
			     row>=(ROCSizeInX*theChipsInX)) {
	  cout<<"PixelIndices: wrong index 3 "<<endl;
	  return -1;
	}
      }

      // Get the 2d ROC coordinate
      int chipX = row / ROCSizeInX; // row index of the chip 0-1
      int chipY = col / ROCSizeInY; // col index of the chip 0-7

      // Get the ROC id from the 2D index
      rocId = rocIndex(chipX,chipY); 
      if(TP_CHECK_LIMITS && (rocId<0 || rocId>=16) ) {
	cout<<"PixelIndices: wrong roc index "<<rocId<<endl;
	return -1;
      }
      // get the local ROC coordinates
      rowROC = (row%ROCSizeInX); // row in chip
      colROC = (col%ROCSizeInY); // col in chip

      if(rocId<8) { // For lower 8 ROCs the coordinates are reversed
	colROC = 51 - colROC;
	rowROC = 79 - rowROC;
      }

      if(TP_CHECK_LIMITS) {
	if(colROC<0||colROC>=ROCSizeInY||rowROC<0||rowROC>=ROCSizeInX) {
	  cout<<"PixelIndices: wrong index "<<colROC<<" "<<rowROC<<endl;
	  return -1;
	}
      }

      return 0;
  }
  //***********************************************************************
  // Calculate a single number ROC index from the 2 ROC indices (coordinates)
  // chipX and chipY.
  // Goes from 0 to 15.
  inline static int rocIndex(const int chipX, const int chipY) {
    int rocId = -1;
    if(TP_CHECK_LIMITS) {
      if(chipX<0 || chipX>=2 ||chipY<0 || chipY>=8) {
	cout<<"PixelChipIndices: wrong index "<<chipX<<" "<<chipY<<endl;
	return -1;
      }
    }
    if(chipX==0) rocId = chipY + 8;  // should be 8-15
    else if(chipX==1) rocId = 7 - chipY; // should be 0-7

    if(TP_CHECK_LIMITS) {
      if(rocId < 0 || rocId >= (maxROCsInX*maxROCsInY) ) {
	cout << "PixelIndices: Error in ROC index " << rocId << endl;
	return -1;
      }
    }
    return rocId;
  }
  //**************************************************************************
  // Calculate the dcol in ROC from the col in ROC frame.
  // dcols go from 0 to 25.
  inline static int DColumn(const int colROC) {
    int dColumnId = (colROC)/2; // double column 0-25
    if(TP_CHECK_LIMITS) {
      if(dColumnId<0 || dColumnId>=26) {
	cout<<"PixelIndices: wrong dcol index  "<<dColumnId<<" "<<colROC<<endl;
	return -1;
      }
    }
    return dColumnId;
  }
  //*************************************************************************
  // Calcuulate the global dcol index within a module
  // Usefull only forin efficiency calculations.  
  inline static int DColumnInModule(const int dcol, const int chipIndex) {
    int dcolInMod = dcol + chipIndex * 26;
    return dcolInMod;
  }


#ifdef OLD
    // Below are the old OBSOLETE reoutines 
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

    chipX = pixX / ROCSizeInX; // row index of the chip 0-1
    chipY = pixY / ROCSizeInY;    // col index of the chip 0-7

    row = (pixX%ROCSizeInX) + 1; // row in chip
    col = (pixY%ROCSizeInY) + 1;    // col in chip

#ifdef TP_CHECK_LIMITS
    if(chipX < 0 || chipX > (theChipsInX-1) ) 
      cout << " PixelChipIndices: Error in chipX " << chipX << endl;
    if(chipY < 0 || chipY > (theChipsInY-1) ) 
      cout << " PixelChipIndices: Error in chipY " << chipY << endl;
    if(row < 1 || row > ROCSizeInX ) 
      cout << " PixelChipIndices: Error in ROC row# " << row << endl;
    if(col < 1 || col > ROCSizeInY ) 
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
    int pixId = row + (col-dColumnId*2+1)*ROCSizeInX; //id in dCol 1-106 
#ifdef TP_CHECK_LIMITS
    if(dColumnId < 1 || dColumnId > DColsPerROC ) 
      cout << " PixelChipIndices: Error in DColumn Id " << dColumnId << endl;
    if(pixId < 1 || pixId > (2*ROCSizeInX) ) 
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
        id > (((maxROCsInX*maxROCsInY)-1)*100 + DColsPerROC) ) 
      cout << " PixelChipIndices: Error in det dcol id " << id << endl;
#endif

    return id;
  }

#endif
  //***********************************************************************
 private:

    int theColsInDet;      // Columns per Det
    int theRowsInDet;      // Rows per Det
    int theChipsInX;       // Chips in det in X (column direction)
    int theChipsInY;       // Chips in det in Y (row direction)
};

#endif




