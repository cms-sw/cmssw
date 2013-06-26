#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/CSCObjects/interface/CSCChannelTranslator.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"


CSCDbStripConditions::CSCDbStripConditions(const edm::ParameterSet & pset)
: CSCStripConditions(),
  theConditions( pset ),
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


float CSCDbStripConditions::gain(const CSCDetId & id, int channel) const
{
  return theConditions.gain(id, channel)  * theGainsConstant;
}


float CSCDbStripConditions::pedestal(const CSCDetId & id, int channel) const
{
  return theConditions.pedestal(id, channel);
}


float CSCDbStripConditions::pedestalSigma(const CSCDetId& id, int channel) const
{
  return theConditions.pedestalSigma(id, channel);
}


void CSCDbStripConditions::crosstalk(const CSCDetId& id, int channel,
			double stripLength, bool leftRight,
			float & capacitive, float & resistive) const
{
  resistive = theConditions.crosstalkIntercept(id, channel, leftRight)
             * theResistiveCrosstalkScaling;
  float slope = theConditions.crosstalkSlope(id, channel, leftRight);
  // ns before the peak where slope is max
  float maxSlopeTime = 60.;
  // some confusion about +/-
  float capacitiveFraction = fabs(slope)*maxSlopeTime;
  // theCapacitiveCrosstalk is the number needed for 100% xtalk, so
  capacitive = theCapacitiveCrosstalk * capacitiveFraction;
}


void CSCDbStripConditions::fetchNoisifier(const CSCDetId & id, int istrip)
{
  std::vector<float> me(12); // buffer for matrix elements
  theConditions.noiseMatrixElements( id, istrip, me ); // fill it

  CSCCorrelatedNoiseMatrix matrix;
  //TODO get the pedestals right
  matrix(2,2) = me[0]; // item.elem33;
  matrix(3,3) = me[3]; // item.elem44;
  matrix(4,4) = me[6]; // item.elem55;
  matrix(5,5) = me[9]; // item.elem66;
  matrix(6,6) = me[11]; // item.elem77;

  if(doCorrelatedNoise_)
  {
    matrix(2,3) = me[1]; // item.elem34;
    matrix(2,4) = me[2]; // item.elem35;
    matrix(3,4) = me[4]; // item.elem45;
    matrix(3,5) = me[5]; // item.elem46;
    matrix(4,5) = me[7]; // item.elem56;
    matrix(4,6) = me[8]; // item.elem57;
    matrix(5,6) = me[10]; // item.elem67;
  }

  // the other diagonal elements can just come from the pedestal sigma
  float sigma = pedestalSigma(id, istrip);
  //@@  float scaVariance = 2 * sigma * sigma;
  //@@ The '2 *' IS strictly correct, but currently the value in the cond db is 2x too large since
  //@@ it is the rms of the distribution of pedestals of all 8 time samples rather than the rms of
  //@@ the average of the first two time samples
  float scaVariance = sigma * sigma;
  matrix(0,0) = matrix(1,1) = matrix(7,7) = scaVariance;

  // unknown neighbors can be the average of the known neighbors
  //float avgNeighbor = (matrix(2,3)+matrix(3,4)+matrix(4,5)+matrix(5,6))/4.;
  //float avg2away = (matrix(2,4)+matrix(3,5)+matrix(4,6))/3.;
  //matrix(0,1) = matrix(1,2) = matrix(6,7) = avgNeighbor;
  //matrix(0,2) = matrix(1,3) = matrix(5,7) = avg2away;

  if(theNoisifier != 0) delete theNoisifier;
  theNoisifier = new CSCCorrelatedNoisifier(matrix);
}

bool CSCDbStripConditions::isInBadChamber( const CSCDetId& id ) const
{
  return theConditions.isInBadChamber( id );
}
