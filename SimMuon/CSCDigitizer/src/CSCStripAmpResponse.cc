
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/CSCDigitizer/src/CSCStripAmpResponse.h"

#include <cmath>
#include <iostream>

CSCStripAmpResponse::CSCStripAmpResponse(int shapingTime, int tailShaping)
    : theShapingTime(shapingTime), theTailShaping(tailShaping) {}

float CSCStripAmpResponse::calculateAmpResponse(float t) const {
  // Local variables
  double t1, t3, t5, t7, cat = 0.;

  // ! CSC Cathode amplifier-shaper response to single e-

  //     Author: S. Durkin Oct. 13, 1995

  //     Routine gives cathode amplifier response to a single drift electron
  //     Positive ion drift collection is included
  //     FLAGS:      itp  amplifier/shaper peaking time
  //     10,20,30,40,50,60,70,80,90,100,150,200,250,300 nsec available
  //                 itl  amplifier tail cancellation
  //                      1 no tail cancellation
  //                      2 conservative tail cancellation
  //                      3 radical tail cancellation(some charge loss)

  //     calculations were done using maple
  //     4-pole semigaussian shaper is assumed

  //     frequency 2tp/27/exp(-3)/(stp/3+1)**3
  //                           |
  //                           \/
  //     time     (t/tp)**3/exp(-3)*exp(-3t/tp)

  //     this time distribution was convoluted with positive ion
  //     drift formula

  //                1/(t0+t)*.05650     note:normalization estimated from K1
  //                                         Gatti N.I.M. 163,82(1979)
  //                where t0=2.1 nsec (GEM gas)

  //            standard tail cancellation has lowest pole removed (exp approx)
  //            using 1 zero and 1 pole
  //            radical tail cancellation has lowest two poles removed (exp
  //            approx) using 2 zeros and 2 poles

  if (theTailShaping != 0 && theTailShaping != 1 && theTailShaping != 2) {
    edm::LogError("CSCStripElectronicsSim") << "Bad TailShaping CSCStripElectronicsSim" << theTailShaping;
    return 0.;
  }

  switch (theShapingTime) {
      /*
        case 10:
    // no tail cancellation, tp= 10,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.3);
                cat = exp(t * -.06523809524) * .2100138871 + exp(t *
                        -.400952381) * 23.38590029 + exp(t * -.01333333333) *
                        .02085199407 + exp(t * -.002285714286) * .004416201989 +
                        t5 * .00345590637 * t * t7 - t5 * .1260070972 * t7 + t *
                         2.304266717 * t7 - t7 * 23.62118237;
            }
    //  tail cancellation, tp= 10,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.3);
                cat = t1 * .003469104089 * t * t3 - t1 * .1263548058 * t3 + t *
                        2.310721678 * t3 - t3 * 23.68962448 + exp(t *
                        -.01333333333) * .02323396152 + exp(t * -.400952381) *
                        23.45252892 + exp(t * -.06523809524) * .2138615931 +
    exp( t * -.003418326524) * 1.274497241e-11;
            }
    //  radical tail cancellation, tp= 10,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.3);
                cat = t1 * .003561830965 * t * t3 - t1 * .1287358512 * t3 + t *
                        2.355429782 * t3 - t3 * 24.16270544 - exp(t *
                        -.0207962661) * 1.717680715e-10 + exp(t *
    -.003418326524)
                         * 7.271661158e-12 + exp(t * -.400952381) * 23.91293094
                        + exp(t * -.06523809524) * .2497744869;
            }
            break;
        case 20:
    // no tail cancellation, tp= 20,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.15);
                cat = exp(t * -.06523809524) * 1.544776642 + exp(t *
                        -.400952381) * .07655350666 + exp(t * -.01333333333) *
                        .0504563677 + exp(t * -.002285714286) * .009108935882 +
                        t5 * 1.849663895e-5 * t * t7 - t5 * .008530427568 * t7 -
                        t * .1199681494 * t7 - t7 * 1.680895453;
            }
    //  tail cancellation, tp= 20,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.15);
                cat = t1 * 1.863955948e-5 * t * t3 - t1 * .008593415697 * t3 -
                        t * .1217545504 * t3 - t3 * 1.706070522 + exp(t *
                        -.01333333333) * .05622010555 + exp(t * -.400952381) *
                        .07677161489 + exp(t * -.06523809524) * 1.573078801 +
                        exp(t * -.003418326524) * 2.669976603e-11;
            }
    //  radical tail cancellation, tp= 20,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.15);
                cat = t1 * 1.971619848e-5 * t * t3 - t1 * .009064781071 * t3 -
                        t * .1360836779 * t3 - t3 * 1.915518161 - exp(t *
                        -.0207962661) * 4.682061154e-10 + exp(t *
    -.003418326524)
                         * 1.523358744e-11 + exp(t * -.400952381) * .07827873625
                        + exp(t * -.06523809524) * 1.837239425;
            }
            break;
    */
    case 30:
      // no tail cancellation, tp= 30,t0=2.1
      if (theTailShaping == NONE) {
        // Computing 2nd power
        t5 = t * t;
        t7 = exp(t * -.1);
        cat = exp(t * -.06523809524) * 16.18007335 + exp(t * -.400952381) * .01096643477 +
              exp(t * -.01333333333) * .0924451733 + exp(t * -.002285714286) * .01409456303 -
              t5 * 7.567748611e-5 * t * t7 - t5 * .01068701841 * t7 - t * .5685389492 * t7 - t7 * 16.29757952;
      }
      //  tail cancellation, tp= 30,t0=2.1
      if (theTailShaping == CONSERVATIVE) {
        // Computing 2nd power
        t1 = t * t;
        t3 = exp(t * -.1);
        cat = t1 * -7.656495508e-5 * t * t3 - t1 * .01083991133 * t3 - t * .5783722846 * t3 - t3 * 16.59051472 +
              exp(t * -.01333333333) * .1030053814 + exp(t * -.400952381) * .01099767919 +
              exp(t * -.06523809524) * 16.47651166 + exp(t * -.003418326524) * 4.197333487e-11;
      }
      //  radical tail cancellation, tp= 30,t0=2.1
      if (theTailShaping == RADICAL) {
        // Computing 2nd power
        t1 = t * t;
        t3 = exp(t * -.1);
        cat = t1 * -8.37792502e-5 * t * t3 - t1 * .0121345525 * t3 - t * .6655605456 * t3 - t3 * 19.25455777 -
              exp(t * -.0207962661) * 9.823832281e-10 + exp(t * -.003418326524) * 2.394794269e-11 +
              exp(t * -.400952381) * .01121357717 + exp(t * -.06523809524) * 19.2433442;
      }
      break;
      /*
        case 40:
    // no tail cancellation, tp= 40,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.075);
                cat = exp(t * -.06523809524) * 1097.590331 + exp(t *
                        -.400952381) * .003362201622 + exp(t * -.01333333333) *
                        .1521508944 + exp(t * -.002285714286) * .01939048554 -
                        t5 * 1.579570388e-4 * t * t7 - t5 * .05281648589 * t7 -
                        t * 10.72426897 * t7 - t7 * 1097.765235;
            }
    //  tail cancellation, tp= 40,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.075);
                cat = t1 * -1.604563386e-4 * t * t3 - t1 * .05375692904 * t3 -
                        t * 10.92023147 * t3 - t3 * 1117.872403 + exp(t *
                        -.01333333333) * .1695314135 + exp(t * -.400952381) *
                        .003371780853 + exp(t * -.06523809524) * 1117.6995 +
    exp( t * -.003418326524) * 5.868522605e-11;
            }
    //  radical tail cancellation, tp= 40,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.075);
                cat = t1 * -1.825484488e-4 * t * t3 - t1 * .06238107065 * t3 -
                        t * 12.7419738 * t3 - t3 * 1305.393597 - exp(t *
                        -.0207962661) * 1.889428819e-9 + exp(t * -.003418326524)
                        * 3.348293469e-11 + exp(t * -.400952381) * .003437973064
                        + exp(t * -.06523809524) * 1305.390159;
            }
            break;
        case 50:
    // no tail cancellation, tp= 50,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.06);
                cat = exp(t * -.06523809524) * 6778.833027 + exp(t *
                        -.400952381) * .001437922352 + exp(t * -.01333333333) *
                        .237530423 + exp(t * -.002285714286) * .02501521924 + t5
                        * 1.670506271e-4 * t * t7 - t5 * .09338148045 * t7 + t *
                         35.49613478 * t7 - t7 * 6779.097011;
            }
    //  tail cancellation, tp= 50,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.06);
                cat = t1 * 1.703945294e-4 * t * t3 - t1 * .09507342939 * t3 + t
                        * 36.14686596 * t3 - t3 * 6903.29548 + exp(t *
                        -.01333333333) * .2646640264 + exp(t * -.400952381) *
                        .001442019131 + exp(t * -.06523809524) * 6903.029373 +
                        exp(t * -.003418326524) * 7.696672009e-11;
            }
    //  radical tail cancellation, tp= 50,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.06);
                cat = t1 * 2.028313098e-4 * t * t3 - t1 * .1106897038 * t3 + t *
                         42.23121193 * t3 - t3 * 8062.228042 - exp(t *
                        -.0207962661) * 3.535147864e-9 + exp(t * -.003418326524)
                        * 4.391346571e-11 + exp(t * -.400952381) * .001470327741
                        + exp(t * -.06523809524) * 8062.226572;
            }
            break;
        case 60:
    // no tail cancellation, tp= 60,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.05);
                cat = exp(t * -.06523809524) * 54.77495457 + exp(t *
                        -.400952381) * 7.412655802e-4 + exp(t * -.01333333333) *
                        .3606757184 + exp(t * -.002285714286) * .03098847247 +
                        t5 * 3.411745868e-5 * t * t7 - t5 * .006682738979 * t7 +
                        t * .8202227541 * t7 - t7 * 55.16736002;
            }
    //  tail cancellation, tp= 60,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.05);
                cat = t1 * 3.494700919e-5 * t * t3 - t1 * .006791801191 * t3 +
                        t * .8354834745 * t3 - t3 * 56.18111721 + exp(t *
                        -.01333333333) * .40187647 + exp(t * -.400952381) *
                        7.433775162e-4 + exp(t * -.06523809524) * 55.77849736 +
                        exp(t * -.003418326524) * 9.696210758e-11;
            }
    //  radical tail cancellation, tp= 60,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.05);
                cat = t1 * 4.387761992e-5 * t * t3 - t1 * .007610015855 * t3 +
                        t * .9929540112 * t3 - t3 * 65.14590792 - exp(t *
                        -.0207962661) * 6.643853476e-9 + exp(t * -.003418326524)
                        * 5.532186095e-11 + exp(t * -.400952381) * 7.57970932e-4
                        + exp(t * -.06523809524) * 65.14514995;
            }
            break;
        case 70:
    // no tail cancellation, tp= 70,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.04285714286);
                cat = exp(t * -.06523809524) * 7.412251365 + exp(t *
                        -.400952381) * 4.306575393e-4 + exp(t * -.01333333333) *
                        .5403469187 + exp(t * -.002285714286) * .03733123063 +
                        t5 * 1.441232259e-5 * t * t7 - t5 * .002150258989 * t7 +
                        t * .1485797804 * t7 - t7 * 7.990360172;
            }
    //  tail cancellation, tp= 70,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.04285714286);
                cat = t1 * 1.48262187e-5 * t * t3 - t1 * .002180526664 * t3 + t
                        * .1513118077 * t3 - t3 * 8.150556465 + exp(t *
                        -.01333333333) * .6020718923 + exp(t * -.400952381) *
                        4.318845237e-4 + exp(t * -.06523809524) * 7.548052688 +
                        exp(t * -.003418326524) * 1.188301202e-10;
            }
    //  radical tail cancellation, tp= 70,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.04285714286);
                cat = t1 * 1.984175252e-5 * t * t3 - t1 * .00223612384 * t3 + t
                        * .1974584549 * t3 - t3 * 8.816006345 - exp(t *
                        -.0207962661) * 1.284819542e-8 + exp(t * -.003418326524)
                        * 6.779868504e-11 + exp(t * -.400952381) * 4.40362949e-4
                        + exp(t * -.06523809524) * 8.815565996;
            }
            break;
        case 80:
    // no tail cancellation, tp= 80,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.0375);
                cat = exp(t * -.06523809524) * 2.104660307 + exp(t *
                        -.400952381) * 2.718693973e-4 + exp(t * -.01333333333) *
                        .806342238 + exp(t * -.002285714286) * .04406584799 + t5
                        * 7.444156851e-6 * t * t7 - t5 * .00109040647 * t7 + t *
                         .03743972817 * t7 - t7 * 2.955340262;
            }
    //  tail cancellation, tp= 80,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.0375);
                cat = t1 * 7.69154327e-6 * t * t3 - t1 * .001104867276 * t3 + t
                        * .0378353391 * t3 - t3 * 3.041945271 + exp(t *
                        -.01333333333) * .8984524213 + exp(t * -.400952381) *
                        2.726439789e-4 + exp(t * -.06523809524) * 2.143220205 +
                        exp(t * -.003418326524) * 1.42745537e-10;
            }
    //  radical tail cancellation, tp= 80,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.0375);
                cat = t1 * 1.112798871e-5 * t * t3 - t1 * 9.813148933e-4 * t3 +
                        t * .06953288418 * t3 - t3 * 2.503400292 - exp(t *
                        -.0207962661) * 2.618804719e-8 + exp(t * -.003418326524)
                        * 8.144365827e-11 + exp(t * -.400952381) *
                        2.779963161e-4 + exp(t * -.06523809524) * 2.50312232;
            }
            break;
        case 90:
    // no tail cancellation, tp= 90,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.03333333333);
                cat = exp(t * -.06523809524) * .8445180788 + exp(t *
                        -.400952381) * 1.824319111e-4 + exp(t * -.01333333333) *
                        1.207282396 + exp(t * -.002285714286) * .051216146 + t5
    * 4.216536082e-6 * t * t7 - t5 * 7.082920848e-4 * t7 + t
                        * .001275426356 * t7 - t7 * 2.103199053;
            }
    //  tail cancellation, tp= 90,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.03333333333);
                cat = t1 * 4.376178381e-6 * t * t3 - t1 * 7.190991486e-4 * t3 +
                        t * 6.011974229e-4 * t3 - t3 * 2.205366435 + exp(t *
                        -.01333333333) * 1.345192821 + exp(t * -.400952381) *
                        1.829516768e-4 + exp(t * -.06523809524) * .8599906622 +
                        exp(t * -.003418326524) * 1.68900989e-10;
            }
    //  radical tail cancellation, tp= 90,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.03333333333);
                cat = t1 * 6.981183555e-6 * t * t3 - t1 * 5.238041184e-4 * t3 +
                        t * .03211389084 * t3 - t3 * 1.004591827 - exp(t *
                        -.0207962661) * 5.795834545e-8 + exp(t * -.003418326524)
                        * 9.636668658e-11 + exp(t * -.400952381) *
                        1.865432436e-4 + exp(t * -.06523809524) * 1.004405341;
            }
            break;
    */
    case 100:
      // no tail cancellation, tp=100,t0=2.1
      if (theTailShaping == NONE) {
        // Computing 2nd power
        t5 = t * t;
        t7 = exp(t * -.03);
        cat = exp(t * -.06523809524) * .4137221868 + exp(t * -.400952381) * 1.282766787e-4 +
              exp(t * -.01333333333) * 1.824993745 + exp(t * -.002285714286) * .05880752038 +
              t5 * 2.491640871e-6 * t * t7 - t5 * 5.417458918e-4 * t7 - t * .01742000448 * t7 - t7 * 2.297651729;
      }
      //  tail cancellation, tp=100,t0=2.1
      if (theTailShaping == CONSERVATIVE) {
        // Computing 2nd power
        t1 = t * t;
        t3 = exp(t * -.03);
        cat = t1 * 2.597806608e-6 * t * t3 - t1 * 5.528471798e-4 * t3 - t * .0189975081 * t3 - t3 * 2.454897362 +
              exp(t * -.01333333333) * 2.033466647 + exp(t * -.400952381) * 1.28642151e-4 +
              exp(t * -.06523809524) * .4213020729 + exp(t * -.003418326524) * 1.975089906e-10;
      }
      //  radical tail cancellation, tp=100,t0=2.1
      if (theTailShaping == RADICAL) {
        // Computing 2nd power
        t1 = t * t;
        t3 = exp(t * -.03);
        cat = t1 * 4.704262123e-6 * t * t3 - t1 * 3.14519427e-4 * t3 + t * .01738754854 * t3 - t3 * .4921806115 -
              exp(t * -.0207962661) * 1.454692205e-7 + exp(t * -.003418326524) * 1.126890204e-10 +
              exp(t * -.400952381) * 1.311675549e-4 + exp(t * -.06523809524) * .4920495894;
      }
      break;
      /*
        case 150:
    // no tail cancellation, tp=150,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.02);
                cat = exp(t * -.06523809524) * .0451302243 + exp(t *
                        -.400952381) * 3.417148182e-5 + exp(t * -.01333333333) *
                        21.12261275 + exp(t * -.002285714286) * .1043948969 - t5
                        * 1.285903907e-7 * t * t7 - t5 * 5.344294733e-4 * t7 -
                        t * .1406120762 * t7 - t7 * 21.27217204;
            }
    //  tail cancellation, tp=150,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.02);
                cat = t1 * -1.373737668e-7 * t * t3 - t1 * 5.725228237e-4 * t3 -
                        t * .1548112258 * t3 - t3 * 23.58148492 + exp(t *
                        -.01333333333) * 23.5354936 + exp(t * -.400952381) *
                        3.426883961e-5 + exp(t * -.06523809524) * .04595706409 +
                        exp(t * -.003418326524) * 3.864816979e-10;
            }
    //  radical tail cancellation, tp=150,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.02);
                cat = t1 * 1.150149576e-6 * t * t3 - t1 * 5.745717877e-5 * t3 +
                        t * .00244082808 * t3 - t3 * .05294004164 - exp(t *
                        -.0207962661) * 7.693441476e-4 + exp(t * -.003418326524)
                        * 2.205076529e-10 + exp(t * -.400952381) *
                        3.494157914e-5 + exp(t * -.06523809524) * .05367444399;
            }
            break;
        case 200:
    // no tail cancellation, tp=200,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.015);
                cat = exp(t * -.06523809524) * .01251802645 + exp(t *
                        -.400952381) * 1.36834457e-5 + exp(t * -.01333333333) *
                        2281.242177 + exp(t * -.002285714286) * .1659547829 - t5
                        * 1.421417147e-6 * t * t7 - t5 * .003198621512 * t7 - t
                        * 3.803546128 * t7 - t7 * 2281.420663;
            }
    //  tail cancellation, tp=200,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.015);
                cat = t1 * -1.560422488e-6 * t * t3 - t1 * .003547432409 * t3 -
                        t * 4.235743145 * t3 - t3 * 2541.846068 + exp(t *
                        -.01333333333) * 2541.833308 + exp(t * -.400952381) *
                        1.372243114e-5 + exp(t * -.06523809524) * .01274737169 +
                        exp(t * -.003418326524) * 6.850793727e-10;
            }
    //  radical tail cancellation, tp=200,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.015);
                cat = t1 * 4.486861205e-7 * t * t3 - t1 * 1.982974483e-5 * t3 +
                        t * 7.533435174e-4 * t3 - t3 * .01490186176 - exp(t *
                        -.0207962661) * 1.155963393e-7 + exp(t * -.003418326524)
                        * 3.908729581e-10 + exp(t * -.400952381) *
                        1.399181936e-5 + exp(t * -.06523809524) * .01488798515;
            }
            break;
        case 250:
    // no tail cancellation, tp=250,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t;
                t7 = exp(t * -.012);
                cat = exp(t * -.06523809524) * .005082161364 + exp(t *
                        -.400952381) * 6.792264583e-6 + exp(t * -.01333333333) *
                        2851.552722 + exp(t * -.002285714286) * .2493354963 + t5
                        * 1.282866561e-6 * t * t7 - t5 * .002554194047 * t7 + t
                        * 3.799921386 * t7 - t7 * 2851.807146;
            }
    //  tail cancellation, tp=250,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.012);
                cat = t1 * 1.452179735e-6 * t * t3 - t1 * .0028321086 * t3 + t *
                         4.236667018 * t3 - t3 * 3177.296817 + exp(t *
                        -.01333333333) * 3177.291635 + exp(t * -.400952381) *
                        6.811616389e-6 + exp(t * -.06523809524) * .005175272648
                        + exp(t * -.003418326524) * 1.163611956e-9;
            }
    //  radical tail cancellation, tp=250,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.012);
                cat = t1 * 2.201206315e-7 * t * t3 - t1 * 9.091071906e-6 * t3 +
                        t * 3.244901734e-4 * t3 - t3 * .006051269645 - exp(t *
                        -.0207962661) * 1.115870681e-8 + exp(t * -.003418326524)
                        * 6.639003673e-10 + exp(t * -.400952381) *
                        6.945336809e-6 + exp(t * -.06523809524) * .006044334803;
            }
            break;
        case 300:
    // no tail cancellation, tp=300,t0=2.1
            if (theTailShaping == NONE) {
    // Computing 2nd power
                t5 = t * t ;
                t7 = exp(t * -.01);
                cat = exp(t * -.06523809524) * .002537698188 + exp(t *
                        -.400952381) * 3.850890276e-6 + exp(t * -.01333333333) *
                        42.24522552 + exp(t * -.002285714286) * .3628281578 + t5
                        * 3.42649397e-7 * t * t7 - t5 * 2.496575507e-4 * t7 + t
                        * .1381601414 * t7 - t7 * 42.61059522;
            }
    //  tail cancellation, tp=300,t0=2.1
            if (theTailShaping == CONSERVATIVE) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.01);
                cat = t1 * 4.016144767e-7 * t * t3 - t1 * 2.657431191e-4 * t3 +
                        t * .1570475462 * t3 - t3 * 47.07357524 + exp(t *
                        -.01333333333) * 47.07098718 + exp(t * -.400952381) *
                        3.861861828e-6 + exp(t * -.06523809524) * .002584191857
                        + exp(t * -.003418326524) * 1.946287754e-9;
            }
    //  radical tail cancellation, tp=300,t0=2.1
            if (theTailShaping == RADICAL) {
    // Computing 2nd power
                t1 = t * t;
                t3 = exp(t * -.01);
                cat = t1 * 1.239979555e-7 * t * t3 - t1 * 4.905476843e-6 * t3 +
                        t * 1.682559598e-4 * t3 - t3 * .00302208046 - exp(t *
                        -.0207962661) * 2.845569802e-9 + exp(t * -.003418326524)
                        * 1.110457097e-9 + exp(t * -.400952381) * 3.937674932e-6
                        + exp(t * -.06523809524) * .00301814452;
            }
            break;
    */
    default:
      edm::LogError("CSCStripElectronicsSim") << "Bad shaping time CSCStripElectronicsSim " << theShapingTime;
      break;
  }
  return cat;
}
