///////////////////////////////////////////////////////////////////////////////
// File: HEDarkening.cc
// Description: simple helper class containing parameterized function 
//              to be used for the SLHC darkening calculation in HE 
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HEDarkening.h"
#include <cmath>

HEDarkening::HEDarkening() {
  // RADIUS (cm)
  for(int i=0;i<24;i++) radius[i]=40+i*10;
  //DOSES for 500fb-1 (Mrad, Mika Huhtinen, CMS IN 2001/050 appro., from 40 up to 70cm)
  //LAYER -1 and 0
  dose_lm1_l0[0 ]=7.00;    dose_lm1_l0[1 ]=3.00;    dose_lm1_l0[2 ]=1.70;    dose_lm1_l0[3 ]=1.20;    dose_lm1_l0[4 ]=1.35;    dose_lm1_l0[5 ]=.811;     
  dose_lm1_l0[6 ]=.656;    dose_lm1_l0[7 ]=.312;    dose_lm1_l0[8 ]=.264;    dose_lm1_l0[9 ]=.231;    dose_lm1_l0[10]=.196;    dose_lm1_l0[11]=.151;     
  dose_lm1_l0[12]=.139;    dose_lm1_l0[13]=.102;    dose_lm1_l0[14]=.0441;   dose_lm1_l0[15]=.0217;   dose_lm1_l0[16]=.0154;   dose_lm1_l0[17]=.00699;     
  dose_lm1_l0[18]=.00324;  dose_lm1_l0[19]=.00202;  dose_lm1_l0[20]=.000967; dose_lm1_l0[21]=.000729; dose_lm1_l0[22]=.000204; dose_lm1_l0[23]=.000117;     
  //LAYER 1 - 3 
  dose_l1_l3[0 ]=10.0;    dose_l1_l3[1 ]=3.00;    dose_l1_l3[2 ]=1.6;    dose_l1_l3[3 ]=1.200;    dose_l1_l3[4 ]=.387;    dose_l1_l3[5 ]=.216;     
  dose_l1_l3[6 ]=.172;    dose_l1_l3[7 ]=.102;    dose_l1_l3[8 ]=.0611;   dose_l1_l3[9 ]=.0515;   dose_l1_l3[10]=.0411;   dose_l1_l3[11]=.0340;     
  dose_l1_l3[12]=.0286;   dose_l1_l3[13]=.0244;   dose_l1_l3[14]=.0170;   dose_l1_l3[15]=.00931;  dose_l1_l3[16]=.00642;  dose_l1_l3[17]=.00417;     
  dose_l1_l3[18]=.00206;  dose_l1_l3[19]=.00162;  dose_l1_l3[20]=.00103;  dose_l1_l3[21]=.000769; dose_l1_l3[22]=.000315; dose_l1_l3[23]=.000231;     
  //LAYER 4 - 5 
  dose_l4_l5[0 ]=8.00;    dose_l4_l5[1 ]=1.30;    dose_l4_l5[2 ]=.700;    dose_l4_l5[3 ]=.500;    dose_l4_l5[4 ]=.152;    dose_l4_l5[5 ]=.086;     
  dose_l4_l5[6 ]=.0666;   dose_l4_l5[7 ]=.0401;   dose_l4_l5[8 ]=.0213;   dose_l4_l5[9 ]=.0172;   dose_l4_l5[10]=.0131;   dose_l4_l5[11]=.0113;     
  dose_l4_l5[12]=.00898;  dose_l4_l5[13]=.00871;  dose_l4_l5[14]=.00740;  dose_l4_l5[15]=.00444;  dose_l4_l5[16]=.00302;  dose_l4_l5[17]=.00190;     
  dose_l4_l5[18]=.000987; dose_l4_l5[19]=.000873; dose_l4_l5[20]=.000490; dose_l4_l5[21]=.000312; dose_l4_l5[22]=.000245; dose_l4_l5[23]=.000131;     
  //LAYER 6 - 8 
  dose_l6_l8[0 ]=8.00;    dose_l6_l8[1 ]=1.00;    dose_l6_l8[2 ]=.500;    dose_l6_l8[3 ]=.300;   dose_l6_l8[4 ]=.0773;    dose_l6_l8[5 ]=.0439;     
  dose_l6_l8[6 ]=.0334;   dose_l6_l8[7 ]=.0194;   dose_l6_l8[8 ]=.0102;   dose_l6_l8[9 ]=.00847;  dose_l6_l8[10]=.00491;   dose_l6_l8[11]=.00507;     
  dose_l6_l8[12]=.00389;  dose_l6_l8[13]=.00296;  dose_l6_l8[14]=.00300;  dose_l6_l8[15]=.00369;  dose_l6_l8[16]=.00219;   dose_l6_l8[17]=.00111;     
  dose_l6_l8[18]=.000543; dose_l6_l8[19]=.000284; dose_l6_l8[20]=.000191; dose_l6_l8[21]=.000165; dose_l6_l8[22]=.0000985; dose_l6_l8[23]=.0000435;     
  //LAYER 9 - 10 
  dose_l9_l10[0 ]=5.93;    dose_l9_l10[1 ]=2.96;    dose_l9_l10[2 ]=.163;     dose_l9_l10[3 ]=.0463;    dose_l9_l10[4 ]=.0292;    dose_l9_l10[5 ]=.0156;     
  dose_l9_l10[6 ]=.0142;   dose_l9_l10[7 ]=.00669;  dose_l9_l10[8 ]=.00415;   dose_l9_l10[9 ]=.00273;   dose_l9_l10[10]=.00177;   dose_l9_l10[11]=.00195;     
  dose_l9_l10[12]=.00152;  dose_l9_l10[13]=.000911; dose_l9_l10[14]=.000894;  dose_l9_l10[15]=.000728;  dose_l9_l10[16]=.000768;  dose_l9_l10[17]=.000410;     
  dose_l9_l10[18]=.000219; dose_l9_l10[19]=.000183; dose_l9_l10[20]=.0000959; dose_l9_l10[21]=.0000836; dose_l9_l10[22]=.0000363; dose_l9_l10[23]=.0000300;     
  //LAYER 11 - 13 
  dose_l11_l13[0 ]=5.17;     dose_l11_l13[1 ]=7.70;     dose_l11_l13[2 ]=.0974;    dose_l11_l13[3 ]=.0301;    dose_l11_l13[4 ]=.0168;    dose_l11_l13[5 ]=.00865;     
  dose_l11_l13[6 ]=.00597;   dose_l11_l13[7 ]=.00320;   dose_l11_l13[8 ]=.00178;   dose_l11_l13[9 ]=.00136;   dose_l11_l13[10]=.000692;  dose_l11_l13[11]=.000684;     
  dose_l11_l13[12]=.000518;  dose_l11_l13[13]=.000414;  dose_l11_l13[14]=.000296;  dose_l11_l13[15]=.000323;  dose_l11_l13[16]=.000346;   dose_l11_l13[17]=.000275;     
  dose_l11_l13[18]=.0000684; dose_l11_l13[19]=.0000523; dose_l11_l13[20]=.0000252; dose_l11_l13[21]=.0000350; dose_l11_l13[22]=.0000226; dose_l11_l13[23]=.0000125;     
  //LAYER 14 - 15 
  dose_l14_l15[0 ]=5.38;     dose_l14_l15[1 ]=2.62;     dose_l14_l15[2 ]=.1000;    dose_l14_l15[3 ]=.0171;    dose_l14_l15[4 ]=.0111;    dose_l14_l15[5 ]=.00420;     
  dose_l14_l15[6 ]=.00334;   dose_l14_l15[7 ]=.00177;   dose_l14_l15[8 ]=.000822;   dose_l14_l15[9 ]=.000626;  dose_l14_l15[10]=.000449;  dose_l14_l15[11]=.000277;     
  dose_l14_l15[12]=.000229;  dose_l14_l15[13]=.000266;  dose_l14_l15[14]=.000129;  dose_l14_l15[15]=.000968;  dose_l14_l15[16]=.000118;  dose_l14_l15[17]=.000104;     
  dose_l14_l15[18]=.0000601; dose_l14_l15[19]=.0000253; dose_l14_l15[20]=.0000267; dose_l14_l15[21]=.0000498; dose_l14_l15[22]=.0000604; dose_l14_l15[23]=.00000281;     
  //LAYER 16 - 17 
  dose_l16_l17[0 ]=5.02;     dose_l16_l17[1 ]=2.55;     dose_l16_l17[2 ]=.222;     dose_l16_l17[3 ]=.0492;    dose_l16_l17[4 ]=.0105;    dose_l16_l17[5 ]=.00434;     
  dose_l16_l17[6 ]=.00293;   dose_l16_l17[7 ]=.00153;   dose_l16_l17[8 ]=.000816;  dose_l16_l17[9 ]=.000683;  dose_l16_l17[10]=.000426;  dose_l16_l17[11]=.000438;     
  dose_l16_l17[12]=.000196;  dose_l16_l17[13]=.000101;  dose_l16_l17[14]=.000162;  dose_l16_l17[15]=.000139;  dose_l16_l17[16]=.0000726; dose_l16_l17[17]=.0000796;     
  dose_l16_l17[18]=.0000684; dose_l16_l17[19]=.0000819; dose_l16_l17[20]=.0000149; dose_l16_l17[21]=.0000376; dose_l16_l17[22]=.0000254; dose_l16_l17[23]=.0000214;     
  
}

HEDarkening::~HEDarkening() { }

float HEDarkening::dose(int layer,float Radius)  {

  //      if((layer<-1)||(layer>17)) return (0.);

  int ind = (Radius-40.)/10.;

  if((layer==-1)||(layer==0)) {
    if(ind<0) return (dose_lm1_l0[0]); if(ind>=23) return (dose_lm1_l0[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_lm1_l0[ind+1]-dose_lm1_l0[ind])/10; return (dose_lm1_l0[ind]+delta*(Radius-radius[ind]));}
  }
  if((layer>=1)&&(layer<=3)) {
    if(ind<0) return (dose_l1_l3[0]); if(ind>=23) return (dose_l1_l3[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l1_l3[ind+1]-dose_l1_l3[ind])/10; return (dose_l1_l3[ind]+delta*(Radius-radius[ind]));}
  }
  if((layer==4)||(layer==5)) {
    if(ind<0) return (dose_l4_l5[0]); if(ind>=23) return (dose_l4_l5[23]);
    if((ind>=0)&&(ind<=23)) {float delta = (dose_l4_l5[ind+1]-dose_l4_l5[ind])/10; return (dose_l4_l5[ind]+delta*(Radius-radius[ind]));}
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

}

float HEDarkening::degradation(float mrad) {
  return (exp(-mrad/6.4));
}

float HEDarkening::int_lumi(float intlumi) {
  return (intlumi/500.);
}
