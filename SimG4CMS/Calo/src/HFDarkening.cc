//////////////////////////////////////////////////////////////////////
//File: HFDarkening.cc
//Description:  simple helper class containing parameterized function
//              to be used for the SLHC darkening calculation in HF
//////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFDarkening.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <cmath>

HFDarkening::HFDarkening() {
  //HF area of consideration is 1100cm from interaction point to 1300cm in z-axis
  //Radius (cm) - 12.5 cm from Beam pipe up to 170cm to top of PMT robox.
  //Dose in MRad
    
  for(int i = 0; i < 10; ++i) radius[i] = 12.5+i*15.75;
  //Depth 0 - 1100cm-1120 - calculated @ 1110cm
  dose_layer_depth[0][0] = 45.8;    dose_layer_depth[0][1] = 39.5;
  dose_layer_depth[0][2] = 9.97;     dose_layer_depth[0][3] = 6.46;
  dose_layer_depth[0][4] = 3.32;    dose_layer_depth[0][5] = 2.21;
  dose_layer_depth[0][6] = 0.737;   dose_layer_depth[0][7] = 0.0557;
  dose_layer_depth[0][8] = 0.00734;   dose_layer_depth[0][9] = 0.00212;
    
  //Depth 1 - 1120-1140 - @1130
  dose_layer_depth[1][0] = 45.8;   dose_layer_depth[1][1] = 39.5;
  dose_layer_depth[1][2] = 9.97;   dose_layer_depth[1][3] = 6.46;
  dose_layer_depth[1][4] = 3.32;   dose_layer_depth[1][5] = 2.21;
  dose_layer_depth[1][6] = 0.737;   dose_layer_depth[1][7] = 0.0557;
  dose_layer_depth[1][8] = 0.00734;   dose_layer_depth[1][9] = 0.00212;
    
  //Depth 2 - 1140-1160 - @1150
  dose_layer_depth[2][0] = 74.8;   dose_layer_depth[2][1] = 42.3;
  dose_layer_depth[2][2] = 8.17;   dose_layer_depth[2][3] = 5.12;
  dose_layer_depth[2][4] = 2.21;   dose_layer_depth[2][5] = 1.48;
  dose_layer_depth[2][6] = 0.400;   dose_layer_depth[2][7] = 0.0388;
  dose_layer_depth[2][8] = 0.0050;   dose_layer_depth[2][9] = 0.00202;
    
  //Depth 3 - 1160-1180 - @1170
  dose_layer_depth[3][0] = 73.9;   dose_layer_depth[3][1] = 18.1;
  dose_layer_depth[3][2] = 3.81;   dose_layer_depth[3][3] = 1.95;
  dose_layer_depth[3][4] = 0.839;   dose_layer_depth[3][5] = 0.559;
  dose_layer_depth[3][6] = 0.200;   dose_layer_depth[3][7] = 0.0386;
  dose_layer_depth[3][8] = 0.00582;   dose_layer_depth[3][9] = 0.00216;
    
  //Depth 4 - 1180-1200 - @1190
  dose_layer_depth[4][0] = 68.0;   dose_layer_depth[4][1] = 9.75;
  dose_layer_depth[4][2] = 2.19;   dose_layer_depth[4][3] = 0.974;
  dose_layer_depth[4][4] = 0.426;   dose_layer_depth[4][5] = 0.265;
  dose_layer_depth[4][6] = 0.124;   dose_layer_depth[4][7] = 0.0258;
  dose_layer_depth[4][8] = 0.00565;   dose_layer_depth[4][9] = 0.00206;
    
  //Depth 5 - 1200-1220 - @1210
  dose_layer_depth[5][0] = 73.1;   dose_layer_depth[5][1] = 5.42;
  dose_layer_depth[5][2] = 1.24;   dose_layer_depth[5][3] = 0.507;
  dose_layer_depth[5][4] = 0.21;   dose_layer_depth[5][5] = 0.129;
  dose_layer_depth[5][6] = 0.0625;   dose_layer_depth[5][7] = 0.0220;
  dose_layer_depth[5][8] = 0.00404;   dose_layer_depth[5][9] = 0.00156;
    
  //Depth 6 - 1220-1240 - @1230
  dose_layer_depth[6][0] = 66.1;   dose_layer_depth[6][1] = 2.79;
  dose_layer_depth[6][2] = 0.650;   dose_layer_depth[6][3] = 0.257;
  dose_layer_depth[6][4] = 0.111;   dose_layer_depth[6][5] = 0.0624;
  dose_layer_depth[6][6] = 0.0335;   dose_layer_depth[6][7] = 0.0134;
  dose_layer_depth[6][8] = 0.00280;   dose_layer_depth[6][9] = 0.00127;
    
  //Depth 7 - 1240-1260 - @1250
  dose_layer_depth[7][0] = 68.7;   dose_layer_depth[7][1] = 1.69;
  dose_layer_depth[7][2] = 0.350;   dose_layer_depth[7][3] = 0.137;
  dose_layer_depth[7][4] = 0.0614;   dose_layer_depth[7][5] = 0.0321;
  dose_layer_depth[7][6] = 0.0167;   dose_layer_depth[7][7] = 0.00621;
  dose_layer_depth[7][8] = 0.00187;   dose_layer_depth[7][9] = 0.000867;
    
  //Depth 8 - 1260-1280 - @1270
  dose_layer_depth[8][0] = 62.5;   dose_layer_depth[8][1] = 2.39;
  dose_layer_depth[8][2] = 0.252;   dose_layer_depth[8][3] = 0.135;
  dose_layer_depth[8][4] = 0.0505;   dose_layer_depth[8][5] = 0.0295;
  dose_layer_depth[8][6] = 0.0159;   dose_layer_depth[8][7] = 0.0131;
  dose_layer_depth[8][8] = 0.00803;   dose_layer_depth[8][9] = 0.00161;
    
  //Depth 9 - 1280-1300 - @1290
  dose_layer_depth[9][0] = 48.6;   dose_layer_depth[9][1] = 204.0;
  dose_layer_depth[9][2] = 0.260;   dose_layer_depth[9][3] = 2.19;
  dose_layer_depth[9][4] = 0.528;   dose_layer_depth[9][5] = 0.386;
  dose_layer_depth[9][6] = 0.231;   dose_layer_depth[9][7] = 0.19;
  dose_layer_depth[9][8] = 0.773;   dose_layer_depth[9][9] = 0.00502;
    
}

HFDarkening::~HFDarkening() { }

float HFDarkening::dose(int layer, float Radius) {
  //	float dose_acquired = 0.;
  int radius = (int) floor((Radius-12.5)/10.0+0.5);
  if (layer == 15) {
    return 0.;
  }
  return dose_layer_depth[layer][radius];

  /*
  for (int i = 0; i<= layer; ++i) {
    dose_acquired+=dose_layer_depth[i][radius];
  }
	
  if((layer<-1)||(layer>17)) return (0.);

  if((layer==-1)||(layer==0)) {
    if(ind<0) return (dose_layer_depth[0][0]); if(ind>=9) return (dose_layer_depth[0][10]);
    if((ind>=0)&&(ind<=9)) {float delta = (dose_layer_depth[0][ind+1]-dose_layer_depth[0][ind])/10; return (dose_layer_depth[0][ind]+delta*(Radius-radius[ind]));}
  }
  if((layer>=1)&&(layer<=3)) {
    if(ind<0) return (dose_layer_depth[5][0]); if(ind>=9) return (dose_layer_depth[5][10]);
    if((ind>=0)&&(ind<=9)) {float delta = (dose_layer_depth[5][ind+1]-dose_layer_depth[5][ind])/10; return (dose_layer_depth[5][ind]+delta*(Radius-radius[ind]));}
  }
  if((layer==4)||(layer==5)) {
    if(ind<0) return (dose_layer_depth[10][0]); if(ind>=9) return (dose_layer_depth[10][10]);
    if((ind>=0)&&(ind<=9)) {float delta = (dose_layer_depth[10][ind+1]-dose_layer_depth[10][ind])/10; return (dose_layer_depth[10][ind]+delta*(Radius-radius[ind]));}
  }
  if((layer>=6)&&(layer<=8)) {
    if(ind<0) return (dose_l6_l8[0]); if(ind>=23) return (dose_l6_l8[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l6_l8[ind+1]-dose_l6_l8[ind])/10; return (dose_l6_l8[ind]+delta*(Radius-radius[ind]));}
  }
  if((layer==9)||(layer==10)) {
    if(ind<0) return (dose_l9_l10[0]); if(ind>=23) return (dose_l9_l10[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l9_l10[ind+1]-dose_l9_l10[ind])/10; return (dose_l9_l10[ind]+delta*(Radius-radius[ind]));}
  }
  if((layer>=11)&&(layer<=13)) {
    if(ind<0) return (dose_l11_l13[0]); if(ind>=23) return (dose_l11_l13[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l11_l13[ind+1]-dose_l11_l13[ind])/10; return (dose_l11_l13[ind]+delta*(Radius-radius[ind]));}
  }
  if((layer==14)||(layer==15)) {
    if(ind<0) return (dose_l14_l15[0]); if(ind>=23) return (dose_l14_l15[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l14_l15[ind+1]-dose_l14_l15[ind])/10; return (dose_l14_l15[ind]+delta*(Radius-radius[ind]));}
  }
  if((layer==16)||(layer==17)) {
    if(ind<0) return (dose_l16_l17[0]); if(ind>=23) return (dose_l16_l17[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l16_l17[ind+1]-dose_l16_l17[ind])/10; return (dose_l16_l17[ind]+delta*(Radius-radius[ind]));}
  }
  return 0.;
  */
}

float HFDarkening::degradation(float mrad) {
  return (exp(-1.44*pow(mrad/100,0.44)*0.2/4.343));
}

float HFDarkening::int_lumi(float intlumi) {
  return (intlumi/500.);
}
