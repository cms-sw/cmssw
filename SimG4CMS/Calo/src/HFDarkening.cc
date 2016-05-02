//////////////////////////////////////////////////////////////////////
//File: HFDarkening.cc
//Description:  simple helper class containing parameterized function
//              to be used for the SLHC darkening calculation in HF.
//              Dose values from FLUKA version 1.0.0.0, contact Sophie Mallows for questions.
//////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFDarkening.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <cmath>

HFDarkening::HFDarkening()
{
  //HF area of consideration is 1115 cm from interaction point to 1280cm in z-axis
  //Radius (cm) - 13 cm from Beam pipe to 130cm (the top of HF active area)
  //Dose in MRad
  
  //Depth 0: 1115. - 1120. cm
  dose_layer_radius[0][0] = 1089.82672;	dose_layer_radius[0][1] = 430.74392;	dose_layer_radius[0][2] = 316.10688;	dose_layer_radius[0][3] = 198.15032;
  dose_layer_radius[0][4] = 133.66563;	dose_layer_radius[0][5] = 92.69111;	dose_layer_radius[0][6] = 63.84813;
  dose_layer_radius[0][7] = 45.19114;	dose_layer_radius[0][8] = 30.58487;	dose_layer_radius[0][9] = 21.00216;
  dose_layer_radius[0][10] = 12.95498;	dose_layer_radius[0][11] = 5.45337;	dose_layer_radius[0][12] = 0.4561;
  
  //Depth 1: 1120. - 1125. cm
  dose_layer_radius[1][0] = 1662.5198;	dose_layer_radius[1][1] = 720.46188;	dose_layer_radius[1][2] = 590.62325;	dose_layer_radius[1][3] = 429.82339;
  dose_layer_radius[1][4] = 282.39827;	dose_layer_radius[1][5] = 186.10415;	dose_layer_radius[1][6] = 120.91404;
  dose_layer_radius[1][7] = 79.7575;	dose_layer_radius[1][8] = 51.45268;	dose_layer_radius[1][9] = 32.77064;
  dose_layer_radius[1][10] = 17.95221;	dose_layer_radius[1][11] = 6.07963;	dose_layer_radius[1][12] = 0.32217;
  
  //Depth 2: 1125. - 1130. cm
  dose_layer_radius[2][0] = 1511.32547;	dose_layer_radius[2][1] = 708.54584;	dose_layer_radius[2][2] = 605.99167;	dose_layer_radius[2][3] = 495.66233;
  dose_layer_radius[2][4] = 322.7595;	dose_layer_radius[2][5] = 207.45045;	dose_layer_radius[2][6] = 129.52382;
  dose_layer_radius[2][7] = 81.14806;	dose_layer_radius[2][8] = 51.51518;	dose_layer_radius[2][9] = 30.54072;
  dose_layer_radius[2][10] = 15.88012;	dose_layer_radius[2][11] = 5.04118;	dose_layer_radius[2][12] = 0.29664;
  
  //Depth 3: 1130. - 1135. cm
  dose_layer_radius[3][0] = 1116.32013;	dose_layer_radius[3][1] = 574.64553;	dose_layer_radius[3][2] = 480.43324;	dose_layer_radius[3][3] = 409.50734;
  dose_layer_radius[3][4] = 266.19544;	dose_layer_radius[3][5] = 168.75067;	dose_layer_radius[3][6] = 104.73339;
  dose_layer_radius[3][7] = 63.34151;	dose_layer_radius[3][8] = 39.86312;	dose_layer_radius[3][9] = 22.97123;
  dose_layer_radius[3][10] = 11.55581;	dose_layer_radius[3][11] = 3.68691;	dose_layer_radius[3][12] = 0.29269;
  
  //Depth 4: 1135. - 1140. cm
  dose_layer_radius[4][0] = 800.2853;	dose_layer_radius[4][1] = 446.24557;	dose_layer_radius[4][2] = 359.63381;	dose_layer_radius[4][3] = 299.49924;
  dose_layer_radius[4][4] = 195.82939;	dose_layer_radius[4][5] = 124.59848;	dose_layer_radius[4][6] = 77.29738;
  dose_layer_radius[4][7] = 46.00855;	dose_layer_radius[4][8] = 28.52664;	dose_layer_radius[4][9] = 16.43682;
  dose_layer_radius[4][10] = 8.18736;	dose_layer_radius[4][11] = 2.67659;	dose_layer_radius[4][12] = 0.28495;
  
  //Depth 5: 1140. - 1145. cm
  dose_layer_radius[5][0] = 612.20865;	dose_layer_radius[5][1] = 355.40823;	dose_layer_radius[5][2] = 276.97155;	dose_layer_radius[5][3] = 219.61391;
  dose_layer_radius[5][4] = 144.74911;	dose_layer_radius[5][5] = 93.28837;	dose_layer_radius[5][6] = 58.51092;
  dose_layer_radius[5][7] = 35.08033;	dose_layer_radius[5][8] = 21.47604;	dose_layer_radius[5][9] = 12.29605;
  dose_layer_radius[5][10] = 6.16789;	dose_layer_radius[5][11] = 2.12159;	dose_layer_radius[5][12] = 0.27661;
  
  //Depth 6: 1145. - 1150. cm
  dose_layer_radius[6][0] = 504.87242;	dose_layer_radius[6][1] = 295.98737;	dose_layer_radius[6][2] = 226.62047;	dose_layer_radius[6][3] = 171.60095;
  dose_layer_radius[6][4] = 113.64104;	dose_layer_radius[6][5] = 73.25579;	dose_layer_radius[6][6] = 46.67008;
  dose_layer_radius[6][7] = 28.26167;	dose_layer_radius[6][8] = 17.42322;	dose_layer_radius[6][9] = 9.59497;
  dose_layer_radius[6][10] = 5.02061;	dose_layer_radius[6][11] = 1.76107;	dose_layer_radius[6][12] = 0.28319;
  
  //Depth 7: 1150. - 1155. cm
  dose_layer_radius[7][0] = 439.39199;	dose_layer_radius[7][1] = 252.97218;	dose_layer_radius[7][2] = 191.68489;	dose_layer_radius[7][3] = 143.625;
  dose_layer_radius[7][4] = 94.50761;	dose_layer_radius[7][5] = 61.46441;	dose_layer_radius[7][6] = 38.8409;
  dose_layer_radius[7][7] = 23.68308;	dose_layer_radius[7][8] = 14.69344;	dose_layer_radius[7][9] = 7.98846;
  dose_layer_radius[7][10] = 4.18968;	dose_layer_radius[7][11] = 1.52224;	dose_layer_radius[7][12] = 0.28134;
  
  //Depth 8: 1155. - 1160. cm
  dose_layer_radius[8][0] = 400.29378;	dose_layer_radius[8][1] = 223.24941;	dose_layer_radius[8][2] = 165.77051;	dose_layer_radius[8][3] = 124.7812;
  dose_layer_radius[8][4] = 80.69381;	dose_layer_radius[8][5] = 52.80131;	dose_layer_radius[8][6] = 33.61851;
  dose_layer_radius[8][7] = 20.37541;	dose_layer_radius[8][8] = 12.53798;	dose_layer_radius[8][9] = 7.00165;
  dose_layer_radius[8][10] = 3.62975;	dose_layer_radius[8][11] = 1.38213;	dose_layer_radius[8][12] = 0.26854;
  
  //Depth 9: 1160. - 1165. cm
  dose_layer_radius[9][0] = 367.42369;	dose_layer_radius[9][1] = 201.03927;	dose_layer_radius[9][2] = 146.05054;	dose_layer_radius[9][3] = 108.58533;
  dose_layer_radius[9][4] = 70.57479;	dose_layer_radius[9][5] = 46.07769;	dose_layer_radius[9][6] = 29.50341;
  dose_layer_radius[9][7] = 17.87488;	dose_layer_radius[9][8] = 10.91036;	dose_layer_radius[9][9] = 6.12909;
  dose_layer_radius[9][10] = 3.21967;	dose_layer_radius[9][11] = 1.25779;	dose_layer_radius[9][12] = 0.25991;
  
  //Depth 10: 1165. - 1170. cm
  dose_layer_radius[10][0] = 329.90783;	dose_layer_radius[10][1] = 181.01783;	dose_layer_radius[10][2] = 128.96837;	dose_layer_radius[10][3] = 94.23766;
  dose_layer_radius[10][4] = 62.25865;	dose_layer_radius[10][5] = 40.17712;	dose_layer_radius[10][6] = 25.787;
  dose_layer_radius[10][7] = 16.16342;	dose_layer_radius[10][8] = 9.46068;	dose_layer_radius[10][9] = 5.41504;
  dose_layer_radius[10][10] = 2.81787;	dose_layer_radius[10][11] = 1.13574;	dose_layer_radius[10][12] = 0.2552;
  
  //Depth 11: 1170. - 1175. cm
  dose_layer_radius[11][0] = 297.89937;	dose_layer_radius[11][1] = 163.50102;	dose_layer_radius[11][2] = 116.23316;	dose_layer_radius[11][3] = 83.18498;
  dose_layer_radius[11][4] = 55.46801;	dose_layer_radius[11][5] = 35.4402;	dose_layer_radius[11][6] = 22.19535;
  dose_layer_radius[11][7] = 14.31524;	dose_layer_radius[11][8] = 8.21573;	dose_layer_radius[11][9] = 4.69969;
  dose_layer_radius[11][10] = 2.50891;	dose_layer_radius[11][11] = 1.02412;	dose_layer_radius[11][12] = 0.24167;
  
  //Depth 12: 1175. - 1180. cm
  dose_layer_radius[12][0] = 272.26937;	dose_layer_radius[12][1] = 147.64059;	dose_layer_radius[12][2] = 104.98641;	dose_layer_radius[12][3] = 72.05083;
  dose_layer_radius[12][4] = 48.82044;	dose_layer_radius[12][5] = 31.0403;	dose_layer_radius[12][6] = 19.50233;
  dose_layer_radius[12][7] = 12.6764;	dose_layer_radius[12][8] = 7.06137;	dose_layer_radius[12][9] = 4.11597;
  dose_layer_radius[12][10] = 2.14441;	dose_layer_radius[12][11] = 0.93751;	dose_layer_radius[12][12] = 0.22596;
  
  //Depth 13: 1180. - 1185. cm
  dose_layer_radius[13][0] = 249.14766;	dose_layer_radius[13][1] = 131.41739;	dose_layer_radius[13][2] = 93.58642;	dose_layer_radius[13][3] = 63.48269;
  dose_layer_radius[13][4] = 42.64957;	dose_layer_radius[13][5] = 27.27858;	dose_layer_radius[13][6] = 17.06007;
  dose_layer_radius[13][7] = 10.93655;	dose_layer_radius[13][8] = 6.0742;	dose_layer_radius[13][9] = 3.64389;
  dose_layer_radius[13][10] = 1.88608;	dose_layer_radius[13][11] = 0.84005;	dose_layer_radius[13][12] = 0.22152;
  
  //Depth 14: 1185. - 1190. cm
  dose_layer_radius[14][0] = 226.99648;	dose_layer_radius[14][1] = 116.79805;	dose_layer_radius[14][2] = 82.04924;	dose_layer_radius[14][3] = 55.57321;
  dose_layer_radius[14][4] = 37.60804;	dose_layer_radius[14][5] = 23.8003;	dose_layer_radius[14][6] = 15.12373;
  dose_layer_radius[14][7] = 9.59352;	dose_layer_radius[14][8] = 5.28989;	dose_layer_radius[14][9] = 3.16053;
  dose_layer_radius[14][10] = 1.65219;	dose_layer_radius[14][11] = 0.74207;	dose_layer_radius[14][12] = 0.20077;
  
  //Depth 15: 1190. - 1195. cm
  dose_layer_radius[15][0] = 207.71168;	dose_layer_radius[15][1] = 102.36457;	dose_layer_radius[15][2] = 72.48674;	dose_layer_radius[15][3] = 48.65921;
  dose_layer_radius[15][4] = 32.76174;	dose_layer_radius[15][5] = 20.67312;	dose_layer_radius[15][6] = 13.34669;
  dose_layer_radius[15][7] = 8.26618;	dose_layer_radius[15][8] = 4.70605;	dose_layer_radius[15][9] = 2.83691;
  dose_layer_radius[15][10] = 1.4537;	dose_layer_radius[15][11] = 0.66819;	dose_layer_radius[15][12] = 0.18359;
  
  //Depth 16: 1195. - 1200. cm
  dose_layer_radius[16][0] = 190.76744;	dose_layer_radius[16][1] = 90.63229;	dose_layer_radius[16][2] = 63.78329;	dose_layer_radius[16][3] = 42.44254;
  dose_layer_radius[16][4] = 29.13414;	dose_layer_radius[16][5] = 18.03128;	dose_layer_radius[16][6] = 11.85671;
  dose_layer_radius[16][7] = 7.31436;	dose_layer_radius[16][8] = 4.02773;	dose_layer_radius[16][9] = 2.45523;
  dose_layer_radius[16][10] = 1.26776;	dose_layer_radius[16][11] = 0.58725;	dose_layer_radius[16][12] = 0.18424;
  
  //Depth 17: 1200. - 1205. cm
  dose_layer_radius[17][0] = 178.21925;	dose_layer_radius[17][1] = 81.88379;	dose_layer_radius[17][2] = 57.03839;	dose_layer_radius[17][3] = 37.22998;
  dose_layer_radius[17][4] = 24.90575;	dose_layer_radius[17][5] = 16.00853;	dose_layer_radius[17][6] = 10.46851;
  dose_layer_radius[17][7] = 6.57621;	dose_layer_radius[17][8] = 3.50808;	dose_layer_radius[17][9] = 2.16873;
  dose_layer_radius[17][10] = 1.12083;	dose_layer_radius[17][11] = 0.51166;	dose_layer_radius[17][12] = 0.16961;
  
  //Depth 18: 1205. - 1210. cm
  dose_layer_radius[18][0] = 169.78037;	dose_layer_radius[18][1] = 73.27996;	dose_layer_radius[18][2] = 50.41326;	dose_layer_radius[18][3] = 32.70998;
  dose_layer_radius[18][4] = 21.81181;	dose_layer_radius[18][5] = 14.35241;	dose_layer_radius[18][6] = 9.10403;
  dose_layer_radius[18][7] = 5.68453;	dose_layer_radius[18][8] = 3.12387;	dose_layer_radius[18][9] = 1.85718;
  dose_layer_radius[18][10] = 1.00079;	dose_layer_radius[18][11] = 0.44896;	dose_layer_radius[18][12] = 0.15281;
  
  //Depth 19: 1210. - 1215. cm
  dose_layer_radius[19][0] = 162.59574;	dose_layer_radius[19][1] = 66.56493;	dose_layer_radius[19][2] = 43.80282;	dose_layer_radius[19][3] = 28.71882;
  dose_layer_radius[19][4] = 18.54598;	dose_layer_radius[19][5] = 12.41703;	dose_layer_radius[19][6] = 7.92044;
  dose_layer_radius[19][7] = 4.83008;	dose_layer_radius[19][8] = 2.7721;	dose_layer_radius[19][9] = 1.5999;
  dose_layer_radius[19][10] = 0.86314;	dose_layer_radius[19][11] = 0.37843;	dose_layer_radius[19][12] = 0.13917;
  
  //Depth 20: 1215. - 1220. cm
  dose_layer_radius[20][0] = 158.18666;	dose_layer_radius[20][1] = 59.65062;	dose_layer_radius[20][2] = 38.26824;	dose_layer_radius[20][3] = 24.43072;
  dose_layer_radius[20][4] = 16.41785;	dose_layer_radius[20][5] = 10.76117;	dose_layer_radius[20][6] = 6.82664;
  dose_layer_radius[20][7] = 4.16347;	dose_layer_radius[20][8] = 2.44142;	dose_layer_radius[20][9] = 1.4081;
  dose_layer_radius[20][10] = 0.76763;	dose_layer_radius[20][11] = 0.34437;	dose_layer_radius[20][12] = 0.13022;
  
  //Depth 21: 1220. - 1225. cm
  dose_layer_radius[21][0] = 150.13577;	dose_layer_radius[21][1] = 53.33958;	dose_layer_radius[21][2] = 34.18336;	dose_layer_radius[21][3] = 21.62898;
  dose_layer_radius[21][4] = 14.27769;	dose_layer_radius[21][5] = 9.37977;	dose_layer_radius[21][6] = 5.89517;
  dose_layer_radius[21][7] = 3.6894;	dose_layer_radius[21][8] = 2.16963;	dose_layer_radius[21][9] = 1.18001;
  dose_layer_radius[21][10] = 0.6684;	dose_layer_radius[21][11] = 0.3026;	dose_layer_radius[21][12] = 0.12473;
  
  //Depth 22: 1225. - 1230. cm
  dose_layer_radius[22][0] = 144.67223;	dose_layer_radius[22][1] = 50.31662;	dose_layer_radius[22][2] = 31.59758;	dose_layer_radius[22][3] = 19.6959;
  dose_layer_radius[22][4] = 12.52437;	dose_layer_radius[22][5] = 8.238;	dose_layer_radius[22][6] = 5.24535;
  dose_layer_radius[22][7] = 3.24359;	dose_layer_radius[22][8] = 1.87348;	dose_layer_radius[22][9] = 1.03871;
  dose_layer_radius[22][10] = 0.58652;	dose_layer_radius[22][11] = 0.26342;	dose_layer_radius[22][12] = 0.11119;
  
  //Depth 23: 1230. - 1235. cm
  dose_layer_radius[23][0] = 136.98616;	dose_layer_radius[23][1] = 45.60196;	dose_layer_radius[23][2] = 28.92017;	dose_layer_radius[23][3] = 17.66014;
  dose_layer_radius[23][4] = 10.97254;	dose_layer_radius[23][5] = 7.36928;	dose_layer_radius[23][6] = 4.53706;
  dose_layer_radius[23][7] = 2.80986;	dose_layer_radius[23][8] = 1.65139;	dose_layer_radius[23][9] = 0.89451;
  dose_layer_radius[23][10] = 0.49702;	dose_layer_radius[23][11] = 0.22609;	dose_layer_radius[23][12] = 0.09944;
  
  //Depth 24: 1235. - 1240. cm
  dose_layer_radius[24][0] = 132.02552;	dose_layer_radius[24][1] = 41.30027;	dose_layer_radius[24][2] = 26.02537;	dose_layer_radius[24][3] = 15.64933;
  dose_layer_radius[24][4] = 9.94368;	dose_layer_radius[24][5] = 6.42028;	dose_layer_radius[24][6] = 3.90554;
  dose_layer_radius[24][7] = 2.47866;	dose_layer_radius[24][8] = 1.47008;	dose_layer_radius[24][9] = 0.8163;
  dose_layer_radius[24][10] = 0.4273;	dose_layer_radius[24][11] = 0.20726;	dose_layer_radius[24][12] = 0.08913;
  
  //Depth 25: 1240. - 1245. cm
  dose_layer_radius[25][0] = 129.36562;	dose_layer_radius[25][1] = 38.47342;	dose_layer_radius[25][2] = 22.91677;	dose_layer_radius[25][3] = 13.76957;
  dose_layer_radius[25][4] = 8.7923;	dose_layer_radius[25][5] = 5.50856;	dose_layer_radius[25][6] = 3.44091;
  dose_layer_radius[25][7] = 2.1008;	dose_layer_radius[25][8] = 1.26223;	dose_layer_radius[25][9] = 0.69379;
  dose_layer_radius[25][10] = 0.38069;	dose_layer_radius[25][11] = 0.17625;	dose_layer_radius[25][12] = 0.07827;
  
  //Depth 26: 1245. - 1250. cm
  dose_layer_radius[26][0] = 127.78974;	dose_layer_radius[26][1] = 35.60432;	dose_layer_radius[26][2] = 20.49112;	dose_layer_radius[26][3] = 12.18895;
  dose_layer_radius[26][4] = 7.72131;	dose_layer_radius[26][5] = 4.74755;	dose_layer_radius[26][6] = 2.91111;
  dose_layer_radius[26][7] = 1.82576;	dose_layer_radius[26][8] = 1.12159;	dose_layer_radius[26][9] = 0.5982;
  dose_layer_radius[26][10] = 0.34274;	dose_layer_radius[26][11] = 0.15086;	dose_layer_radius[26][12] = 0.06907;
  
  //Depth 27: 1250. - 1255. cm
  dose_layer_radius[27][0] = 124.07525;	dose_layer_radius[27][1] = 33.24689;	dose_layer_radius[27][2] = 18.53004;	dose_layer_radius[27][3] = 10.8278;
  dose_layer_radius[27][4] = 6.97957;	dose_layer_radius[27][5] = 4.0935;	dose_layer_radius[27][6] = 2.46634;
  dose_layer_radius[27][7] = 1.63918;	dose_layer_radius[27][8] = 0.97005;	dose_layer_radius[27][9] = 0.50387;
  dose_layer_radius[27][10] = 0.29183;	dose_layer_radius[27][11] = 0.13397;	dose_layer_radius[27][12] = 0.05972;
  
  //Depth 28: 1255. - 1260. cm
  dose_layer_radius[28][0] = 122.43579;	dose_layer_radius[28][1] = 31.75832;	dose_layer_radius[28][2] = 17.29619;	dose_layer_radius[28][3] = 9.83185;
  dose_layer_radius[28][4] = 6.05158;	dose_layer_radius[28][5] = 3.55086;	dose_layer_radius[28][6] = 2.21287;
  dose_layer_radius[28][7] = 1.39251;	dose_layer_radius[28][8] = 0.82526;	dose_layer_radius[28][9] = 0.45851;
  dose_layer_radius[28][10] = 0.24181;	dose_layer_radius[28][11] = 0.11202;	dose_layer_radius[28][12] = 0.05703;
  
  //Depth 29: 1260. - 1265. cm
  dose_layer_radius[29][0] = 121.61748;	dose_layer_radius[29][1] = 30.44129;	dose_layer_radius[29][2] = 16.17414;	dose_layer_radius[29][3] = 8.93093;
  dose_layer_radius[29][4] = 5.44004;	dose_layer_radius[29][5] = 3.08939;	dose_layer_radius[29][6] = 1.905;
  dose_layer_radius[29][7] = 1.21964;	dose_layer_radius[29][8] = 0.71349;	dose_layer_radius[29][9] = 0.40864;
  dose_layer_radius[29][10] = 0.21309;	dose_layer_radius[29][11] = 0.09492;	dose_layer_radius[29][12] = 0.0535;
  
  //Depth 30: 1265. - 1270. cm
  dose_layer_radius[30][0] = 120.67237;	dose_layer_radius[30][1] = 29.01728;	dose_layer_radius[30][2] = 14.54675;	dose_layer_radius[30][3] = 8.14094;
  dose_layer_radius[30][4] = 4.8913;	dose_layer_radius[30][5] = 2.67395;	dose_layer_radius[30][6] = 1.62092;
  dose_layer_radius[30][7] = 1.04203;	dose_layer_radius[30][8] = 0.59052;	dose_layer_radius[30][9] = 0.32811;
  dose_layer_radius[30][10] = 0.19279;	dose_layer_radius[30][11] = 0.08357;	dose_layer_radius[30][12] = 0.05398;
  
  //Depth 31: 1270. - 1275. cm
  dose_layer_radius[31][0] = 119.15188;	dose_layer_radius[31][1] = 27.50844;	dose_layer_radius[31][2] = 13.95703;	dose_layer_radius[31][3] = 7.2369;
  dose_layer_radius[31][4] = 4.21802;	dose_layer_radius[31][5] = 2.32933;	dose_layer_radius[31][6] = 1.45072;
  dose_layer_radius[31][7] = 0.87999;	dose_layer_radius[31][8] = 0.5075;	dose_layer_radius[31][9] = 0.29475;
  dose_layer_radius[31][10] = 0.16882;	dose_layer_radius[31][11] = 0.08048;	dose_layer_radius[31][12] = 0.05269;
  
  //Depth 32: 1275. - 1280. cm
  dose_layer_radius[32][0] = 114.6991;	dose_layer_radius[32][1] = 25.85671;	dose_layer_radius[32][2] = 12.93505;	dose_layer_radius[32][3] = 6.70319;
  dose_layer_radius[32][4] = 3.86928;	dose_layer_radius[32][5] = 2.3702;	dose_layer_radius[32][6] = 1.37139;
  dose_layer_radius[32][7] = 0.87629;	dose_layer_radius[32][8] = 0.50304;	dose_layer_radius[32][9] = 0.29778;
  dose_layer_radius[32][10] = 0.18413;	dose_layer_radius[32][11] = 0.11134;	dose_layer_radius[32][12] = 0.07111;
  
}

HFDarkening::~HFDarkening()
{}

float HFDarkening::dose(int layer, float Radius)
{
  // Radii are 13-17, 17-20, 20-24, 24-29, 29-34, 34-41, 41-48, 48-58, 58-69, 69-82, 82-98, 98-116, 116-130
  
  if (layer < 0 || layer > 32) {
    return 0.;
  }
  
  int radius = 0;
  if (Radius > 13.0 && Radius <= 17.0) radius = 0;
  if (Radius > 17.0 && Radius <= 20.0) radius = 1;
  if (Radius > 20.0 && Radius <= 24.0) radius = 2;
  if (Radius > 24.0 && Radius <= 29.0) radius = 3;
  if (Radius > 29.0 && Radius <= 34.0) radius = 4;
  if (Radius > 34.0 && Radius <= 41.0) radius = 5;
  if (Radius > 41.0 && Radius <= 48.0) radius = 6;
  if (Radius > 48.0 && Radius <= 58.0) radius = 7;
  if (Radius > 58.0 && Radius <= 69.0) radius = 8;
  if (Radius > 69.0 && Radius <= 82.0) radius = 9;
  if (Radius > 82.0 && Radius <= 98.0) radius = 10;
  if (Radius > 98.0 && Radius <= 116.0) radius = 11;
  if (Radius > 116.0 && Radius <= 130.0) radius = 12;
  if (Radius > 130.0) return 0.;
  
  
  return dose_layer_radius[layer][radius];
}

float HFDarkening::degradation(float mrad)
{
  return (exp(-1.44*pow(mrad/100,0.44)*0.2/4.343));
}

float HFDarkening::int_lumi(float intlumi)
{
  return (intlumi/3000.);
}
