#include "SimMuon/CSCDigitizer/src/CSCDriftSim.h"
#include <cmath>

double CSCDriftSim::avgPathLengthHighB() {
  /* Initialized data */

  static const double coeff[18] = {.16916627,
                                   .11057547,
                                   .054287448,
                                   .01179527,
                                   .0062073273,
                                   -.013570915,
                                   -.0027121772,
                                   -.0053792764,
                                   -.0027452986,
                                   -.0020556715,
                                   .0021511659,
                                   .0011376412,
                                   .0026183373,
                                   .0017980602,
                                   -.0012975418,
                                   -.0010798782,
                                   -.0012322628,
                                   -8.3635924e-4};

  /* Local variables */
  double x10, x11, x12, x13, x14, x20, x21, x22, x23, x24, x25;

  /* ! Parameterization of drift path length - high field chambers */
  /* ***********************************************************************
   */
  /* DOC MC_BHGH_SLEN                                                      *
   */
  /*                                                                      *
   */
  /* DOC  Function    : Parameterization of the drift path length          *
   */
  /* DOC                in the muon endcap CSCs.                           *
   */
  /*                                                                      *
   */
  /* DOC  References  : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Arguments   : YCELL - distance from the anode wire in the        *
   */
  /*                           anode-cathode coordinate plane             *
   */
  /* DOC                ZCELL - distance from the anode wire in the wire   *
   */
  /*                           plane coordinate                           *
   */
  /* DOC  Errors      : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Returns     : Drift path length for high field CSC chambers      *
   */
  /*                                                                      *
   */
  /* DOC  Created     : 15-OCT-1996   Author : Jeff Rowe                   *
   */
  /* ***********************************************************************
   */

  x10 = 1.;
  x11 = fabs(ycell) * 2. - 1.;
  x12 = x11 * 2. * x11 - x10;
  x13 = x11 * 2. * x12 - x11;
  x14 = x11 * 2. * x13 - x12;
  x20 = 1.;
  x21 = fabs(zcell) * 2. - 1.;
  x22 = x21 * 2. * x21 - x20;
  x23 = x21 * 2. * x22 - x21;
  x24 = x21 * 2. * x23 - x22;
  x25 = x21 * 2. * x24 - x23;

  return coeff[0] + coeff[1] * x11 + coeff[2] * x21 + coeff[3] * x22 + coeff[4] * x12 + coeff[5] * x11 * x21 +
         coeff[6] * x13 + coeff[7] * x12 * x22 + coeff[8] * x12 * x23 + coeff[9] * x11 * x24 + coeff[10] * x12 * x21 +
         coeff[11] * x14 + coeff[12] * x11 * x22 + coeff[13] * x13 * x22 + coeff[14] * x13 * x21 +
         coeff[15] * x12 * x24 + coeff[16] * x11 * x25 + coeff[17] * x11 * x23;
}

double CSCDriftSim::pathSigmaHighB() {
  /* Initialized data */

  static const double coeff[9] = {.0049089564,
                                  .0091482062,
                                  .0024036507,
                                  .0065285652,
                                  .0041487742,
                                  -.0038102526,
                                  -.0043923587,
                                  .0019230151,
                                  .0013543258};

  /* System generated locals */
  float ret_val;

  /* Local variables */
  double /*x10,*/ x11, x12, x13, x14, x15, x16, /*x20,*/ x21, x22, x23, x24, x25, x26, x27, x28, x29;

  /* ! Parameterization of path length dispersion- high field chambers */
  /* ***********************************************************************
   */
  /* DOC MC_BHGH_SSIG                                                      *
   */
  /*                                                                      *
   */
  /* DOC  Function    : Parameterization of the drift path length          *
   */
  /* DOC                dispersion in the muon endcap CSCs.                *
   */
  /*                                                                      *
   */
  /* DOC  References  : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Arguments   : YCELL - distance from the anode wire in the        *
   */
  /*                           anode-cathode coordinate plane             *
   */
  /* DOC                ZCELL - distance from the anode wire in the wire   *
   */
  /*                           plane coordinate                           *
   */
  /*           **NOTE** Both distances normalize to cell dim=1x1          *
   */
  /* DOC  Errors      : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Returns     : Path length dispersion for high field CSC chambers *
   */
  /*                                                                      *
   */
  /* DOC  Created     : 15-OCT-1996   Author : Jeff Rowe                   *
   */
  /* ***********************************************************************
   */

  // x10 = 1.; //not used later
  x11 = fabs(ycell) * 2. - 1.;
  x12 = x11 * x11;
  x13 = x11 * x12;
  x14 = x11 * x13;
  x15 = x11 * x14;
  x16 = x11 * x15;
  // x20 = 1.; //not used later
  x21 = fabs(zcell) * 2. - 1.;
  x22 = x21 * x21;
  x23 = x21 * x22;
  x24 = x21 * x23;
  x25 = x21 * x24;
  x26 = x21 * x25;
  x27 = x21 * x26;
  x28 = x21 * x27;
  x29 = x21 * x28;

  ret_val = coeff[0] + coeff[1] * x21 + coeff[2] * x11 + coeff[3] * x22 + coeff[4] * x11 * x21 + coeff[5] * x16 * x22 +
            coeff[6] * x16 * x23 + coeff[7] * x11 * x22 + coeff[8] * x29;

  return ret_val;
}

double CSCDriftSim::avgDriftTimeHighB() {
  /* Initialized data */

  static const double coeff[27] = {22.384492,  10.562894,  14.032961,  7.06233,   3.5523289,  -5.0176704, 1.999075,
                                   1.0635552,  -3.2770096, -2.7384958, .98411495, -2.0963696, -1.4006525, -.47542728,
                                   .64179451,  -.80308436, .42964647,  -.4153324, .50423068,  .35049792,  -.42595896,
                                   -.30947641, .16671267,  -.21336584, .22979164, .23481052,  .32550435};

  /* System generated locals */
  float ret_val;

  /* Local variables */
  double x10, x11, x12, x13, x14, x15, x16, x17, x20, x21, x22, x23, x24, x25, x26, x27;

  /* ! Parameterization of drift time - high field chambers */
  /* ***********************************************************************
   */
  /* DOC MC_BHGH_TIME                                                      *
   */
  /*                                                                      *
   */
  /* DOC  Function    : Parameterization of the drift time                 *
   */
  /* DOC                in the muon endcap CSCs.                           *
   */
  /*                                                                      *
   */
  /* DOC  References  : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Arguments   : YCELL - distance from the anode wire in the        *
   */
  /*                           anode-cathode coordinate plane             *
   */
  /*                           (ycell=1 > d_acat)                         *
   */
  /* DOC                ZCELL - distance from the anode wire in the wire   *
   */
  /*                           plane coordinate (zcell=1 > d_anod/2.)     *
   */
  /* DOC  Errors      : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Returns     : Drift time for high field CSC chambers             *
   */
  /*                                                                      *
   */
  /* DOC  Created     : 15-OCT-1996   Author : Jeff Rowe                   *
   */
  /* ***********************************************************************
   */

  x10 = 1.;
  x11 = fabs(ycell) * 2. - 1.;
  x12 = x11 * 2. * x11 - x10;
  x13 = x11 * 2. * x12 - x11;
  x14 = x11 * 2. * x13 - x12;
  x15 = x11 * 2. * x14 - x13;
  x16 = x11 * 2. * x15 - x14;
  x17 = x11 * 2. * x16 - x15;
  x20 = 1.;
  x21 = fabs(zcell) * 2. - 1.;
  x22 = x21 * 2. * x21 - x20;
  x23 = x21 * 2. * x22 - x21;
  x24 = x21 * 2. * x23 - x22;
  x25 = x21 * 2. * x24 - x23;
  x26 = x21 * 2. * x25 - x24;
  x27 = x21 * 2. * x26 - x25;

  ret_val = coeff[0] + coeff[1] * x11 + coeff[2] * x21 + coeff[3] * x22 + coeff[4] * x23 + coeff[5] * x11 * x21 +
            coeff[6] * x24 + coeff[7] * x12 + coeff[8] * x11 * x22 + coeff[9] * x11 * x23 + coeff[10] * x25 +
            coeff[11] * x11 * x24 + coeff[12] * x11 * x25 + coeff[13] * x13 + coeff[14] * x12 * x21 +
            coeff[15] * x11 * x26 + coeff[16] * x26 + coeff[17] * x11 * x27 + coeff[18] * x17 * x21 +
            coeff[19] * x15 * x21 + coeff[20] * x12 * x22 + coeff[21] * x12 * x23 + coeff[22] * x27 +
            coeff[23] * x14 * x22 + coeff[24] * x16 * x21 + coeff[25] * x17 + coeff[26] * x17 * x22;

  return ret_val;
}

double CSCDriftSim::driftTimeSigmaHighB() {
  /* Initialized data */

  static const double coeff[17] = {5.5533465,
                                   3.3733352,
                                   3.776603,
                                   2.2673355,
                                   1.3401485,
                                   .84209333,
                                   -.71621378,
                                   .57572407,
                                   -.52313936,
                                   -.78790514,
                                   -.71786066,
                                   .43370011,
                                   .29223306,
                                   -.37791975,
                                   .21121024,
                                   .31513644,
                                   .25382701};

  /* System generated locals */
  float ret_val;

  /* Local variables */
  double x10, x11, x12, x13, x14, x15, x16, x17, x18, /*x19,*/ x20, x21, x22, x23, x24, x25, x26, x27, x28, x29;

  /* ! Parameterization of drift time dispersion- high field chambers */
  /* ***********************************************************************
   */
  /* DOC MC_BHGH_TSIG                                                      *
   */
  /*                                                                      *
   */
  /* DOC  Function    : Parameterization of the drift time dispersion      *
   */
  /* DOC                in the muon endcap CSCs.                           *
   */
  /*                                                                      *
   */
  /* DOC  References  : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Arguments   : YCELL - distance from the anode wire in the        *
   */
  /*                           anode-cathode coordinate plane             *
   */
  /*                           (ycell=1 > d_acat)                         *
   */
  /* DOC                ZCELL - distance from the anode wire in the wire   *
   */
  /*                           plane coordinate (zcell=1 > d_anod/2.)     *
   */
  /* DOC  Errors      : None                                               *
   */
  /*                                                                      *
   */
  /* DOC  Returns     : Drift time dispersion for high field CSC chambers  *
   */
  /*                                                                      *
   */
  /* DOC  Created     : 15-OCT-1996   Author : Jeff Rowe                   *
   */
  /* ***********************************************************************
   */

  x10 = 1.;
  x11 = fabs(ycell) * 2. - 1.;
  x12 = x11 * 2. * x11 - x10;
  x13 = x11 * 2. * x12 - x11;
  x14 = x11 * 2. * x13 - x12;
  x15 = x11 * 2. * x14 - x13;
  x16 = x11 * 2. * x15 - x14;
  x17 = x11 * 2. * x16 - x15;
  x18 = x11 * 2. * x17 - x16;
  // x19 = x11 * 2. * x18 - x17; //not used later
  x20 = 1.;
  x21 = fabs(zcell) * 2. - 1.;
  x22 = x21 * 2. * x21 - x20;
  x23 = x21 * 2. * x22 - x21;
  x24 = x21 * 2. * x23 - x22;
  x25 = x21 * 2. * x24 - x23;
  x26 = x21 * 2. * x25 - x24;
  x27 = x21 * 2. * x26 - x25;
  x28 = x21 * 2. * x27 - x26;
  x29 = x21 * 2. * x28 - x27;

  ret_val = coeff[0] * x21 + coeff[1] + coeff[2] * x22 + coeff[3] * x23 + coeff[4] * x24 + coeff[5] * x25 +
            coeff[6] * x11 * x23 + coeff[7] * x26 + coeff[8] * x11 * x25 + coeff[9] * x11 * x24 +
            coeff[10] * x11 * x22 + coeff[11] * x27 + coeff[12] * x28 + coeff[13] * x11 * x26 + coeff[14] * x29 +
            coeff[15] * x16 * x21 + coeff[16] * x18 * x21;

  return ret_val;
}
