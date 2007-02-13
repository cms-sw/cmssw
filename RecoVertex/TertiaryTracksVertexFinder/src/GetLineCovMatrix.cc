
#include "RecoVertex/TertiaryTracksVertexFinder/interface/GetLineCovMatrix.h"

GetLineCovMatrix::GetLineCovMatrix(GlobalPoint pointOne, GlobalPoint pointTwo, GlobalError ErrorOne, GlobalError ErrorTwo)
{

  PointOne = pointOne;
  PointTwo = pointTwo;

  CombinedErrorMatrix = HepMatrix(6, 6, 0);
 
  CombinedErrorMatrix[0][0] = ErrorOne.cxx();
  CombinedErrorMatrix[1][0] = ErrorOne.cyx(); 
  CombinedErrorMatrix[0][1] = ErrorOne.cyx();
  CombinedErrorMatrix[1][1] = ErrorOne.cyy();
  CombinedErrorMatrix[2][0] = ErrorOne.czx();
  CombinedErrorMatrix[0][2] = ErrorOne.czx(); 
  CombinedErrorMatrix[2][1] = ErrorOne.czy();
  CombinedErrorMatrix[1][2] = ErrorOne.czy(); 
  CombinedErrorMatrix[2][2] = ErrorOne.czz();
    
  CombinedErrorMatrix[3][3] = ErrorTwo.cxx();
  CombinedErrorMatrix[4][3] = ErrorTwo.cyx(); 
  CombinedErrorMatrix[3][4] = ErrorTwo.cyx();
  CombinedErrorMatrix[4][4] = ErrorTwo.cyy();
  CombinedErrorMatrix[5][3] = ErrorTwo.czx();
  CombinedErrorMatrix[3][5] = ErrorTwo.czx(); 
  CombinedErrorMatrix[5][4] = ErrorTwo.czy();
  CombinedErrorMatrix[4][5] = ErrorTwo.czy(); 
  CombinedErrorMatrix[5][5] = ErrorTwo.czz();

  B = HepMatrix(3, 6, 0);  
}


GlobalError GetLineCovMatrix::GetMatrix(GlobalPoint PointThree) 
{ 
  // the linear equation is  K = PointOne + (PointTwo-PointOne)*s
  double s;  
  if( !fabs(PointTwo.x() - PointOne.x()) < 0.00000001 ) 
    s =  (PointThree.x() - PointOne.x()) / (PointTwo.x() -  PointOne.x()) ; 
  else {
    if( !fabs(PointTwo.y() - PointOne.y()) < 0.00000001 ) 
      s =  (PointThree.y() - PointOne.y()) / (PointTwo.y() -  PointOne.y()) ;   
    else { 
      if( !fabs(PointTwo.z() - PointOne.z()) < 0.00000001 ) 
	s =  (PointThree.z() - PointOne.z()) / (PointTwo.z() -  PointOne.z()) ;
      else {
        GlobalError EmptyError(0, 0, 0, 0, 0, 0);
        return EmptyError;
      }
    }
  }

  B[0][0] = 1-s;    
  B[0][3] = s;
  B[1][1] = 1-s;   
  B[1][4] = s;
  B[2][2] = 1-s;  
  B[2][5] = s;

  HepMatrix Result = B * CombinedErrorMatrix * B.T();
  
  GlobalError TheGlobalError( Result[0][0],  Result[1][0],  Result[1][1], Result[2][0],  Result[2][1],  Result[2][2] );
  return TheGlobalError;
}

