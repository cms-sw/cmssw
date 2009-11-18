#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

#include <iostream>
#include <iomanip>
#include <fstream>

EcalShape::EcalShape(double timePhase)
{
  setTpeak(timePhase);
  
  nsamp = 250 * 2;

  threshold = 0.00013;

  // tabulated shape from external input in 1 ns bins
  // from 0. to 250. (i.e. 10 x 25 ns ) ns
  // TProfile SHAPE_XTAL_704 from A. Zabi
  // F. Cossutti - 15-Jun-2006 16:03

  // Modififed by Alex Zabi-29/06/06
  // expanding the shape to simulate out-of-time pile-up

  std::vector<double> shapeArray(nsamp,0.0);

  shapeArray[0] = 6.94068e-05 ; 
  shapeArray[1] = -5.03304e-05 ; 
  shapeArray[2] = -2.13404e-05 ; 
  shapeArray[3] = 6.017e-05 ; 
  shapeArray[4] = 2.01697e-05 ; 
  shapeArray[5] = 0.000114845 ; 
  shapeArray[6] = 2.13998e-05 ; 
  shapeArray[7] = 2.74476e-05 ; 
  shapeArray[8] = 5.2824e-05 ; 
  shapeArray[9] = 8.754e-05 ; 
  shapeArray[10] = 2.95346e-06 ; 
  shapeArray[11] = -7.58699e-05 ; 
  shapeArray[12] = -2.72224e-05 ; 
  shapeArray[13] = 3.10997e-06 ; 
  shapeArray[14] = -3.97771e-05 ; 
  shapeArray[15] = -1.06916e-05 ; 
  shapeArray[16] = -0.000113865 ; 
  shapeArray[17] = 6.05044e-05 ; 
  shapeArray[18] = -5.81202e-05 ; 
  shapeArray[19] = -6.58974e-06 ; 
  shapeArray[20] = 5.37494e-05 ; 
  shapeArray[21] = -0.000123729 ; 
  shapeArray[22] = 7.50938e-06 ; 
  shapeArray[23] = -1.35628e-05 ; 
  shapeArray[24] = 8.33725e-05 ; 
  shapeArray[25] = 3.19299e-05 ; 
  shapeArray[26] = -3.09232e-05 ; 
  shapeArray[27] = -7.0086e-05 ; 
  shapeArray[28] = 1.78937e-06 ; 
  shapeArray[29] = -2.20365e-05 ; 
  shapeArray[30] = 7.68054e-05 ; 
  shapeArray[31] = -2.5368e-05 ; 
  shapeArray[32] = 5.67291e-06 ; 
  shapeArray[33] = 5.87096e-05 ; 
  shapeArray[34] = -2.62771e-06 ; 
  shapeArray[35] = 4.31832e-05 ; 
  shapeArray[36] = 8.33616e-06 ; 
  shapeArray[37] = 7.27813e-05 ; 
  shapeArray[38] = 7.6159e-05 ; 
  shapeArray[39] = -1.60446e-05 ; 
  shapeArray[40] = -4.12127e-06 ; 
  shapeArray[41] = -5.93381e-05 ; 
  shapeArray[42] = 1.61444e-05 ; 
  shapeArray[43] = -5.49559e-05 ; 
  shapeArray[44] = 5.55254e-05 ; 
  shapeArray[45] = 3.32251e-05 ; 
  shapeArray[46] = -3.15897e-05 ; 
  shapeArray[47] = 7.86588e-05 ; 
  shapeArray[48] = -2.9704e-05 ; 
  shapeArray[49] = 5.66838e-05 ; 
  shapeArray[50] = 2.85281e-05 ; 
  shapeArray[51] = -3.02436e-05 ; 
  shapeArray[52] = -4.16265e-05 ; 
  shapeArray[53] = -1.63191e-05 ; 
  shapeArray[54] = 6.61193e-05 ; 
  shapeArray[55] = 9.23766e-05 ; 
  shapeArray[56] = 6.68903e-05 ; 
  shapeArray[57] = -3.20994e-05 ; 
  shapeArray[58] = 0.00011082 ; 
  shapeArray[59] = -4.07997e-05 ; 
  shapeArray[60] = -8.29046e-06 ; 
  shapeArray[61] = -7.42197e-05 ; 
  shapeArray[62] = -1.64386e-05 ; 
  shapeArray[63] = 1.02508e-05 ; 
  shapeArray[64] = 7.10995e-06 ; 
  shapeArray[65] = -5.87486e-05 ; 
  shapeArray[66] = -0.000101201 ; 
  shapeArray[67] = 1.62003e-05 ; 
  shapeArray[68] = -2.53093e-05 ; 
  shapeArray[69] = 2.65239e-05 ; 
  shapeArray[70] = -2.68722e-05 ; 
  shapeArray[71] = -4.02001e-05 ; 
  shapeArray[72] = 5.0674e-05 ; 
  shapeArray[73] = -1.75884e-05 ; 
  shapeArray[74] = 4.7902e-05 ; 
  shapeArray[75] = -1.01079e-05 ; 
  shapeArray[76] = 1.08427e-05 ; 
  shapeArray[77] = -0.000112906 ; 
  shapeArray[78] = 3.33076e-05 ; 
  shapeArray[79] = 0.000181201 ; 
  shapeArray[80] = 0.000426875 ; 
  shapeArray[81] = 0.00114222 ; 
  shapeArray[82] = 0.00237804 ; 
  shapeArray[83] = 0.00541858 ; 
  shapeArray[84] = 0.0089021 ; 
  shapeArray[85] = 0.0149157 ; 
  shapeArray[86] = 0.0231397 ; 
  shapeArray[87] = 0.0344671 ; 
  shapeArray[88] = 0.0471013 ; 
  shapeArray[89] = 0.0625517 ; 
  shapeArray[90] = 0.0857351 ; 
  shapeArray[91] = 0.108561 ; 
  shapeArray[92] = 0.133481 ; 
  shapeArray[93] = 0.163557 ; 
  shapeArray[94] = 0.200243 ; 
  shapeArray[95] = 0.225919 ; 
  shapeArray[96] = 0.269213 ; 
  shapeArray[97] = 0.302929 ; 
  shapeArray[98] = 0.342722 ; 
  shapeArray[99] = 0.378522 ; 
  shapeArray[100] = 0.436563 ; 
  shapeArray[101] = 0.467581 ; 
  shapeArray[102] = 0.510133 ; 
  shapeArray[103] = 0.550063 ; 
  shapeArray[104] = 0.583509 ; 
  shapeArray[105] = 0.619187 ; 
  shapeArray[106] = 0.653245 ; 
  shapeArray[107] = 0.686101 ; 
  shapeArray[108] = 0.721178 ; 
  shapeArray[109] = 0.745129 ; 
  shapeArray[110] = 0.774163 ; 
  shapeArray[111] = 0.799011 ; 
  shapeArray[112] = 0.822177 ; 
  shapeArray[113] = 0.838315 ; 
  shapeArray[114] = 0.858847 ; 
  shapeArray[115] = 0.875559 ; 
  shapeArray[116] = 0.891294 ; 
  shapeArray[117] = 0.90537 ; 
  shapeArray[118] = 0.919617 ; 
  shapeArray[119] = 0.930632 ; 
  shapeArray[120] = 0.936216 ; 
  shapeArray[121] = 0.947739 ; 
  shapeArray[122] = 0.955306 ; 
  shapeArray[123] = 0.961876 ; 
  shapeArray[124] = 0.968124 ; 
  shapeArray[125] = 0.97327 ; 
  shapeArray[126] = 0.977513 ; 
  shapeArray[127] = 0.984885 ; 
  shapeArray[128] = 0.986497 ; 
  shapeArray[129] = 0.990039 ; 
  shapeArray[130] = 0.994798 ; 
  shapeArray[131] = 0.994884 ; 
  shapeArray[132] = 0.99795 ; 
  shapeArray[133] = 0.99834 ; 
  shapeArray[134] = 0.999607 ; 
  shapeArray[135] = 1 ; 
  shapeArray[136] = 0.999047 ; 
  shapeArray[137] = 0.998745 ; 
  shapeArray[138] = 0.999219 ; 
  shapeArray[139] = 0.99814 ; 
  shapeArray[140] = 0.995082 ; 
  shapeArray[141] = 0.992449 ; 
  shapeArray[142] = 0.990418 ; 
  shapeArray[143] = 0.985032 ; 
  shapeArray[144] = 0.982308 ; 
  shapeArray[145] = 0.978696 ; 
  shapeArray[146] = 0.975656 ; 
  shapeArray[147] = 0.971027 ; 
  shapeArray[148] = 0.964811 ; 
  shapeArray[149] = 0.959428 ; 
  shapeArray[150] = 0.95096 ; 
  shapeArray[151] = 0.947428 ; 
  shapeArray[152] = 0.9419 ; 
  shapeArray[153] = 0.933223 ; 
  shapeArray[154] = 0.926482 ; 
  shapeArray[155] = 0.922172 ; 
  shapeArray[156] = 0.912777 ; 
  shapeArray[157] = 0.907388 ; 
  shapeArray[158] = 0.897289 ; 
  shapeArray[159] = 0.891889 ; 
  shapeArray[160] = 0.882056 ; 
  shapeArray[161] = 0.873382 ; 
  shapeArray[162] = 0.865442 ; 
  shapeArray[163] = 0.860032 ; 
  shapeArray[164] = 0.85202 ; 
  shapeArray[165] = 0.841013 ; 
  shapeArray[166] = 0.833802 ; 
  shapeArray[167] = 0.825259 ; 
  shapeArray[168] = 0.815013 ; 
  shapeArray[169] = 0.807465 ; 
  shapeArray[170] = 0.799428 ; 
  shapeArray[171] = 0.792165 ; 
  shapeArray[172] = 0.783088 ; 
  shapeArray[173] = 0.773392 ; 
  shapeArray[174] = 0.764982 ; 
  shapeArray[175] = 0.752174 ; 
  shapeArray[176] = 0.746487 ; 
  shapeArray[177] = 0.737678 ; 
  shapeArray[178] = 0.727396 ; 
  shapeArray[179] = 0.718692 ; 
  shapeArray[180] = 0.712737 ; 
  shapeArray[181] = 0.702738 ; 
  shapeArray[182] = 0.69559 ; 
  shapeArray[183] = 0.684389 ; 
  shapeArray[184] = 0.677989 ; 
  shapeArray[185] = 0.667643 ; 
  shapeArray[186] = 0.659009 ; 
  shapeArray[187] = 0.650217 ; 
  shapeArray[188] = 0.644479 ; 
  shapeArray[189] = 0.636017 ; 
  shapeArray[190] = 0.625257 ; 
  shapeArray[191] = 0.618507 ; 
  shapeArray[192] = 0.609798 ; 
  shapeArray[193] = 0.600097 ; 
  shapeArray[194] = 0.592788 ; 
  shapeArray[195] = 0.584895 ; 
  shapeArray[196] = 0.578228 ; 
  shapeArray[197] = 0.569299 ; 
  shapeArray[198] = 0.560576 ; 
  shapeArray[199] = 0.552404 ; 
  shapeArray[200] = 0.541405 ; 
  shapeArray[201] = 0.536271 ; 
  shapeArray[202] = 0.528734 ; 
  shapeArray[203] = 0.519813 ; 
  shapeArray[204] = 0.512264 ; 
  shapeArray[205] = 0.507001 ; 
  shapeArray[206] = 0.49828 ; 
  shapeArray[207] = 0.492416 ; 
  shapeArray[208] = 0.483181 ; 
  shapeArray[209] = 0.477907 ; 
  shapeArray[210] = 0.469623 ; 
  shapeArray[211] = 0.462528 ; 
  shapeArray[212] = 0.455099 ; 
  shapeArray[213] = 0.45055 ; 
  shapeArray[214] = 0.443576 ; 
  shapeArray[215] = 0.435364 ; 
  shapeArray[216] = 0.429789 ; 
  shapeArray[217] = 0.422724 ; 
  shapeArray[218] = 0.415621 ; 
  shapeArray[219] = 0.409469 ; 
  shapeArray[220] = 0.40401 ; 
  shapeArray[221] = 0.398121 ; 
  shapeArray[222] = 0.391079 ; 
  shapeArray[223] = 0.384414 ; 
  shapeArray[224] = 0.378214 ; 
  shapeArray[225] = 0.369851 ; 
  shapeArray[226] = 0.365966 ; 
  shapeArray[227] = 0.359865 ; 
  shapeArray[228] = 0.353505 ; 
  shapeArray[229] = 0.347899 ; 
  shapeArray[230] = 0.343829 ; 
  shapeArray[231] = 0.337585 ; 
  shapeArray[232] = 0.333089 ; 
  shapeArray[233] = 0.326289 ; 
  shapeArray[234] = 0.322249 ; 
  shapeArray[235] = 0.316079 ; 
  shapeArray[236] = 0.31061 ; 
  shapeArray[237] = 0.305426 ; 
  shapeArray[238] = 0.301885 ; 
  shapeArray[239] = 0.296753 ; 
  shapeArray[240] = 0.290931 ; 
  shapeArray[241] = 0.286877 ; 
  shapeArray[242] = 0.281831 ; 
  shapeArray[243] = 0.276633 ; 
  shapeArray[244] = 0.272283 ; 
  shapeArray[245] = 0.268069 ; 
  shapeArray[246] = 0.26399 ; 
  shapeArray[247] = 0.258457 ; 
  shapeArray[248] = 0.253549 ; 
  shapeArray[249] = 0.249493 ; 

  // Alex 29/06/06 - modification to expand the shape up to more the 1500 bins 
  // to simulate out-of-time pile-up properly
  // Exponential fit on the tail of the pulse shape [180,250ns]
  for ( int i = 250 ; i < nsamp; ++i ) shapeArray[i] = exp(2.39735 - 0.0151053* ((double)i+1.0));

  for ( int i = 0 ; i < nsamp; ++i ) {
    LogDebug("EcalShape") << " time (ns) = " << (double)i << " tabulated ECAL pulse shape = " << shapeArray[i];
  }
  
  // first create pulse shape over a range of time 0 ns to 250 ns in 1 ns steps
  // tconv give integer fraction of 1 ns
  tconv = 10;

  nbin = nsamp*tconv;

  std::vector<double> ntmp(nbin,0.0);  // zero output pulse shape
  std::vector<double> ntmpd(nbin,0.0);  // zero output derivative pulse shape

  int j;
  double xb;
  double value,deriv;

  double delta = (1./(double)tconv)/2.;

  for(j=0;j<nbin;j++){
    xb = ((double)(j+1))/tconv-delta; 
    value = 0.0;
    deriv = 0.0;

    unsigned int ibin = j/(int)tconv;

    // A la H4SIM, parabolic interpolation and analytic continuation
    if (ibin < 0 ) { value = 0.; deriv = 0.; }
    else if (ibin == 0) { value = shapeArray[ibin]; deriv = 0.; }
    else if (ibin+1 == shapeArray.size()) { value = shapeArray[ibin]; deriv = 0.;}
    else {
      double x = xb - ((double)ibin+0.5);
      double f1 = shapeArray[ibin - 1];
      double f2 = shapeArray[ibin];
      double f3 = shapeArray[ibin + 1];
      double a = f2;
      double b = (f3 - f1)/2.;
      double c = (f1 + f3)/2. - f2;
      value = a + b*x + c*x*x;
      deriv = (b + 2*c*x)/delta;
    }

    ntmp[j] = value;
    ntmpd[j] = deriv;
  }
  
  for( int i = 0; i < nbin; i++){
    LogDebug("EcalShape") << " time (ns) = " << (double)(i+1)/tconv-delta << " interpolated ECAL pulse shape = " << ntmp[i] << " derivative = " << ntmpd[i];
  }

  nt = ntmp;
  ntd = ntmpd;

  /// Consistency check on the peak time and calculation of the bin where rise starts

  binstart = 0;
  double risingTime = computeRisingTime();

  if ( fabs(risingTime - timePhase) > 1./(double)tconv ) {
    throw(std::runtime_error("EcalShape: computed rising time does not match with input"));
  }

  double T0 = computeT0();
  binstart = (int)(T0*tconv);
}//constructor



double EcalShape::operator () (double time_) const
{

  // return pulse amplitude for request time in ns
  int jtime;
  jtime = (int)(time_*tconv+0.5);
  if(jtime>=0 && jtime<nbin-binstart){
    return nt[jtime+binstart];
  }
  else if (jtime<0) {return 0.;}
  else {    
    LogDebug("EcalShape") << " ECAL MGPA shape requested for out of range time " << time_;
    return 0.0;
  }

}


double EcalShape::derivative (double time_) const
{

  // return pulse amplitude for request time in ns
  int jtime;
  jtime = (int)(time_*tconv+0.5);
  if(jtime>=0 && jtime<nbin-binstart){
    return ntd[jtime+binstart];
  }   
  else if (jtime<0) {return 0.;}
  else {
    LogDebug("EcalShape") << " ECAL MGPA shape derivative requested for out of range time " << time_;
    return 0.0;
  }

}

double EcalShape::computeTimeOfMaximum() const {

  int imax = 0;
  double tmax = -999.;
  for ( int i = 0; i < nsamp*tconv; i++ ) {
    if ( nt[i] >= tmax ) {
      imax = i;
      tmax = nt[i];
    }
  }

  double thePeak = (double)imax/(double)tconv;
  return thePeak;

}

double EcalShape::computeT0() const {

  int istart = 0;
  int i = 1;

  while ( istart == 0 && i < nsamp*tconv ) {

    if (nt[i] >threshold && i > 0) istart = i-1;
    
    ++i;
  }
  
  double theT0 = (double)istart/(double)tconv;
  return theT0;

}

double EcalShape::computeRisingTime() const {

  double ToM = computeTimeOfMaximum();
  double T0 = computeT0();
  return ToM-T0;

}

// Alex Zabi 20/07/07 
// Adding member function load which can be used to load a 
// specific profile. The signal output profile of a given
// crystal can be loaded instead of using the reference
// signal representation described above. 
// WARNING: This function is meant to be used only for test
// beam studies. Specific profiles can be loaded to compute
// optimized weights for resolution studies. 
//NOTE: The format of the profile.txt file is as follows:

// Number of profiles in the file
// xtal number
// bin 1 (starting from 1)  value  error
// bin 2  value error
// bin 3 value error
// ...
// ...
// bin 250 value error
// xtal number
// bin 1  value error
// bin 2  value error
// ...
// ...
// ....

// There are 250 bins in a profile. 1bin = 1ns timing interval.
// 10 samples x 25 TDC (test beam trigger) bins of 1ns  

void EcalShape::load(int xtal_, int SuperModule_)
{

  std::cout << "LOADING NEW XTAL SHAPE" << std::endl;
  //clearing vectors
  for(unsigned int l=0; l < nt.size(); ++l)
    {nt[l] = 0.0; ntd[l] = 0.0;}

  char profile_file[250];
  std::sprintf (profile_file,"profile_SM%02u.txt",SuperModule_);
  std::cout << "LOOKING for " << profile_file << " FILE" << std::endl;
  std::ifstream profile(profile_file);
  int nProfile = 0;
  profile >> nProfile;
  std::cout << "There are " << nProfile 
	    << " Xtal available in the file" << std::endl;
  
  std::vector<double> ProfileData;
  std::vector<int>    Xtal;
  for(int ite=0; ite < nProfile; ++ite)
    {
      int xtalin;
      profile >> xtalin;
      Xtal.push_back(xtalin);
      for(int nb=0; nb < 250; ++nb){
	int    bini  = 0;
	double val   = 0.0;
	double error = 0.0;
	profile >> bini >> val >> error;
	ProfileData.push_back(val);
      }//loop bin
    }//loop profile
  
  int index = 0;
  bool found = false;
  for(int it=0; it < nProfile; ++it)
    if(Xtal[it] == xtal_){
      index = it; found = true;
      std::cout << "XTAL SELECTED=" << Xtal[it] << " index=" << index << std::endl;
    }//check xtal

  if(!found) std::cout << "No XTAL=" << xtal_ << " HERE, taking default" << std::endl;
  else {
    std::vector<double> shapeArray(500,0.0);
    for(int nb=0; nb < 250; ++nb){
      shapeArray[nb] = ProfileData[index*250+nb];      
      //std::cout << nb << " " << ProfileData[index*250+nb] 
      //<< " " << shapeArray[nb] << std::endl;
    }//loop bin
    
    std::vector<double> ntmp(500,0.0);   // zero output pulse shape
    std::vector<double> ntmpd(500,0.0);  // zero output derivative pulse shape
    
    int j;
    double xb;
    double value,deriv;
    
    double delta = (1./(double)tconv)/2.;
    
    for(j=0;j<nbin;j++){
      xb = ((double)(j+1))/tconv-delta; 
      value = 0.0;
      deriv = 0.0;
      
      unsigned int ibin = j/(int)tconv;
      
      // A la H4SIM, parabolic interpolation and analytic continuation
      if (ibin < 0 ) { value = 0.; deriv = 0.; }
      else if (ibin == 0) { value = shapeArray[ibin]; deriv = 0.; }
      else if (ibin+1 == shapeArray.size()) { value = shapeArray[ibin]; deriv = 0.;}
      else {
	double x = xb - ((double)ibin+0.5);
	double f1 = shapeArray[ibin - 1];
	double f2 = shapeArray[ibin];
	double f3 = shapeArray[ibin + 1];
	double a = f2;
	double b = (f3 - f1)/2.;
	double c = (f1 + f3)/2. - f2;
	value = a + b*x + c*x*x;
	deriv = (b + 2*c*x)/delta;
      }
      nt[j]  = value;
      ntd[j] = deriv;
    }

    binstart = 0;
    double T0 = computeT0();
    binstart  = (int)(T0*tconv);
  }//loading shape

  profile.close();
}//loading new shape

const std::vector<double>& EcalShape::getTimeTable() const{

  return nt;

}

const std::vector<double>& EcalShape::getDerivTable() const{

  return ntd;

}
