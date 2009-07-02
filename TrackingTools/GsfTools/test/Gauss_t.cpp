#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/GsfTools/interface/SingleGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"

#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>
#include<vector>


void st(){}
void en(){}

int main(int argc, char * argv[]) {

 
  MultiGaussianState1D::SingleState1dContainer v(6);
  v[0] = SingleGaussianState1DGS(0.,1.,1.);
  v[1] = SingleGaussianState1DGS(0.,2.,0.5);
  v[2] = SingleGaussianState1DGS(0.2,2.,0.5);
  v[3] = SingleGaussianState1DGS(-0.2,2.,0.5);
  v[4] = SingleGaussianState1DGS(0.,4.,1.0);
  v[5] = SingleGaussianState1DGS(0.1,4.,0.3);

  MultiGaussianState1D mgs(v);

  // call once to inizialite compiler stuff
  {
    GaussianSumUtilities1D gsu1(mgs);
    gsu1.mode();
  }

  GaussianSumUtilities1D gsu(mgs);
  st();
  const SingleGaussianState1D sg1 & gsumode();
  end();

  std::cout << sg1.mean() 
	    << " "<< sg1.standardDeviation() 
	    << " " << sg1.weight()
	    << std::endl;


  return 0;
}
