#ifndef LUT_H
#define LUT_H

#include <vector>

class LUT{

 
 public:  

      LUT(): columns(0), rows(0){}
      LUT( int nColumns, int nRows ): columns( nColumns ), rows( nRows ){}
      
      void loadLUT( TString filename );


      int getColumns(){ return columns; }
      int getRows(){    return rows; }

      // Return a specific LUT element. Note: numbering starts from 0 
      double getElement( int row, int column ){
	int index = column + row*columns;
	return LUTArr[ index ];
      }


 private:
      std::vector <double> LUTArr;

      // LUT dimensions 
      int columns;
      int rows;


};




// Read a LUT stored in a textfile and store in a flat vector
void 
LUT::loadLUT( TString filename ){
  
  std::ifstream iReadLUT( filename );

  if(!iReadLUT){
    throw cms::Exception("MissingLUT")  << "A LUT could not be loaded as the file: '" << filename << "' does not exist, check the path of all LUTs.\n";
  } 

  // Read and store the LUT in a flat vector
  while (!iReadLUT.eof()){
    TString element;
    iReadLUT >> element;
    if (element != ""){
      LUTArr.push_back( element.Atof() );
    }
  }

  // Validate input
  if ( int( LUTArr.size() ) != columns*rows ){

    throw cms::Exception("InconsistentLUTEntries")  << "LUT dimensions specified:\n" << "\tRows = " << rows << "\n\tColumns = " << columns 
						    << "\n\nExpected " << rows*columns << " entries in LUT but read " << LUTArr.size() 
						    << " entries, check that the input LUT file contains the same dimensions as the LUT object.\n";

  }

}



#endif

