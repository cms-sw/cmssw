#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"


CSCDbStripConditions::CSCDbStripConditions(const edm::ParameterSet & pset) 
: CSCStripConditions(),
  theConditions(),
  theCapacitiveCrosstalk(pset.getParameter<double>("capacativeCrosstalk")),
  theResistiveCrosstalkScaling(pset.getParameter<double>("resistiveCrosstalkScaling")),
  theGainsConstant(pset.getParameter<double>("gainsConstant")),
  doCorrelatedNoise_(pset.getParameter<bool>("doCorrelatedNoise"))
{
//  theCapacitiveCrosstalk = = 1/maxslope/maxsignal) = 1/ (0.00231/0.143);
// Howoever, need a bit more.  Maybe the slope gets smeared?
}


CSCDbStripConditions::~CSCDbStripConditions()
{
  if(theNoisifier != 0) delete theNoisifier;
}


void CSCDbStripConditions::initializeEvent(const edm::EventSetup & es)
{
  theConditions.initializeEvent(es);
}


float CSCDbStripConditions::gain(const CSCDetId & detId, int channel) const
{
  return theConditions.gain(detId, channel) * theGainsConstant;
}



float CSCDbStripConditions::pedestal(const CSCDetId & detId, int channel) const
{
  return  theConditions.pedestal(detId, channel);
}


float CSCDbStripConditions::pedestalSigma(const CSCDetId&detId, int channel) const
{
  return  theConditions.pedestalSigma(detId, channel);
}


void CSCDbStripConditions::crosstalk(const CSCDetId&detId, int channel,
                 double stripLength, bool leftRight,
                 float & capacitive, float & resistive) const
{
  resistive = theConditions.crosstalkIntercept(detId, channel, leftRight)
             * theResistiveCrosstalkScaling;
  float slope = theConditions.crosstalkSlope(detId, channel, leftRight);
  // ns before the peak where slope is max
  float maxSlopeTime = 60.; 
  // some confusion about +/-
  float capacitiveFraction = fabs(slope)*maxSlopeTime;
  // theCapacitiveCrosstalk is the number needed for 100% xtalk, so
  capacitive = theCapacitiveCrosstalk * capacitiveFraction;
}



void CSCDbStripConditions::fetchNoisifier(const CSCDetId & detId, int istrip)
{
  const CSCDBNoiseMatrix::Item & item = theConditions.noiseMatrix(detId, istrip);

  CSCCorrelatedNoiseMatrix matrix;
  //TODO get the pedestals right
  matrix(3,3) = item.elem33;
  matrix(4,4) = item.elem44;
  matrix(5,5) = item.elem55;
  matrix(6,6) = item.elem66;
  matrix(7,7) = item.elem77;
  
  if(doCorrelatedNoise_)
  {
    matrix(3,4) = item.elem34;
    matrix(3,5) = item.elem35;
    matrix(4,5) = item.elem45;
    matrix(4,6) = item.elem46;
    matrix(5,6) = item.elem56;
    matrix(5,7) = item.elem57;
    matrix(6,7) = item.elem67;
  }

  // the other diagonal elements can just come from the pedestal sigma, I guess
  float sigma = pedestalSigma(detId, istrip);
  float scaVariance = 2 * sigma * sigma;
  matrix(0,0) = matrix(1,1) = matrix(2,2) = scaVariance;
  if(theNoisifier != 0) delete theNoisifier;
  theNoisifier = new CSCCorrelatedNoisifier(matrix);
}

