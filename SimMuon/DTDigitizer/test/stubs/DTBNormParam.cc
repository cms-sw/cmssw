//-------------------------------------------------
//
//   Class: DTBNormParam.cc
//
//   Description:
//       Provide time_vs_x correction functions for different normal B field
//       values
//
//
//   Author List:
//   P. Ronchese           INFN - PADOVA
//   Modifications:
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DTBNormParam.h"

//-------------
// C Headers --
//-------------
#include <cmath>

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------

DTBNormParam::DTBNormParam()
    :  // default constructor : assume 0 field
      _bnorm(0.0),
      l_func(0),
      h_func(1) {}

DTBNormParam::DTBNormParam(float bnorm) {
  // determine Bnorm bin in 0.0/1.0 range by steps of 0.1

  float x_bno = fabs(bnorm) * 10.0;
  int i_bno = static_cast<int>(x_bno);
  int l_bin;

  if (i_bno > 9)
    l_bin = 9;
  else
    l_bin = i_bno;

  int h_bin = l_bin + 1;

  // store actual Bnorm and function parameters for lower/higher bin bounds
  _bnorm = bnorm;
  l_func = ParamFunc(l_bin);
  h_func = ParamFunc(h_bin);
}

//--------------
// Destructor --
//--------------

DTBNormParam::~DTBNormParam() {}

//--------------
// Operations --
//--------------

float DTBNormParam::tcor(float xpos) const {
  // compute drift time for lower/higher bin bounds
  float l_tcor = l_func.tcor(xpos);
  float h_tcor = h_func.tcor(xpos);

  // interpolate time corrections for actual Bnorm
  return l_tcor + ((h_tcor - l_tcor) * l_func.dist(_bnorm) / l_func.dist(h_func));
}

const float DTBNormParam::table_offsc[11] = {0.0,
                                             0.51630E-03,
                                             0.31628E-02,
                                             0.49082E-02,
                                             0.75994E-02,
                                             0.10643E-01,
                                             0.13419E-01,
                                             0.16636E-01,
                                             0.20108E-01,
                                             0.24405E-01,
                                             0.28550E-01};

const float DTBNormParam::table_coeff[11] = {0.0,
                                             0.13900E-03,
                                             -0.16361E-02,
                                             -0.25234E-02,
                                             -0.39586E-02,
                                             -0.56774E-02,
                                             -0.70548E-02,
                                             -0.86567E-02,
                                             -0.10549E-01,
                                             -0.12765E-01,
                                             -0.14833E-01};

DTBNormParam::ParamFunc::ParamFunc() {
  /*
   // default constructor: assume central bin ( 0 field )
   bin_bnorm = 0.0;
   offsc     = table_offsc;
   coeff     = table_coeff;
  */
}

DTBNormParam::ParamFunc::ParamFunc(int bin) {
  // store bound
  bin_bnorm = bin * 0.1;
  // select parameters inside tables
  offsc = table_offsc + bin;
  coeff = table_coeff + bin;
}

/*
void DTBNormParam::ParamFunc::set(int bin) {

 // reset bound
 bin_bnorm = bin*0.1;
 // select parameters inside tables
 offsc     = table_offsc+bin;
 coeff     = table_coeff+bin;

}
*/

DTBNormParam::ParamFunc::~ParamFunc() {}

float DTBNormParam::ParamFunc::tcor(float xpos) const {
  // compute drift time correction
  return *offsc + (*coeff * xpos);
}
