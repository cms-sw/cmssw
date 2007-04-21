#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"


CSCDbStripConditions::CSCDbStripConditions() 
: CSCStripConditions(),
  theNoiseMatrix(0),
  theGains(0),
  thePedestals(0),
  theCrosstalk(0),
  theCapacitiveCrosstalk(5.344),
  theResistiveCrosstalk(0.02)
{
}


CSCDbStripConditions::~CSCDbStripConditions()
{
  delete theNoiseMatrix;
  delete theGains;
  delete thePedestals;
  delete theCrosstalk;
}


void CSCDbStripConditions::initializeEvent(const edm::EventSetup & es)
{
  // Strip gains
  edm::ESHandle<CSCGains> hGains;
  es.get<CSCGainsRcd>().get( hGains );
  theGains = &*hGains.product();
  // Strip X-talk
  edm::ESHandle<CSCcrosstalk> hCrosstalk;
  es.get<CSCcrosstalkRcd>().get( hCrosstalk );
  theCrosstalk = &*hCrosstalk.product();
  // Strip pedestals
  edm::ESHandle<CSCPedestals> hPedestals;
  es.get<CSCPedestalsRcd>().get( hPedestals );
  thePedestals = &*hPedestals.product();

  // Strip autocorrelation noise matrix
  edm::ESHandle<CSCNoiseMatrix> hNoiseMatrix;
  es.get<CSCNoiseMatrixRcd>().get(hNoiseMatrix);
  theNoiseMatrix = &*hNoiseMatrix.product();
}


void CSCDbStripConditions::print() const
{
  std::cout << "SIZES: GAINS: " << theGains->gains.size()
            << "   PEDESTALS: " << thePedestals->pedestals.size()
            << "   NOISES "  << theNoiseMatrix->matrix.size() << std::endl;;

  std::map< int,std::vector<CSCGains::Item> >::const_iterator layerGainsItr = theGains->gains.begin(), 
      lastGain = theGains->gains.end();
  for( ; layerGainsItr != lastGain; ++layerGainsItr)
  {
    std::cout << "GAIN " << layerGainsItr->first << " " << layerGainsItr->second[0].gain_slope 
    << " " << layerGainsItr->second[0].gain_intercept << std::endl;
  }

  std::map< int,std::vector<CSCPedestals::Item> >::const_iterator pedestalItr = thePedestals->pedestals.begin(), 
                                                                  lastPedestal = thePedestals->pedestals.end();
  for( ; pedestalItr != lastPedestal; ++pedestalItr)
  {
    std::cout << "PEDS " << pedestalItr->first << " " << pedestalItr->second[0].ped << " " 
              << pedestalItr->second[0].rms << std::endl;
  }

  std::map< int,std::vector<CSCcrosstalk::Item> >::const_iterator crosstalkItr = theCrosstalk->crosstalk.begin(),
                                                                  lastCrosstalk = theCrosstalk->crosstalk.end();
  for( ; crosstalkItr != lastCrosstalk; ++crosstalkItr)
  {
    std::cout << "XTALKS " << crosstalkItr->first << " " 
     << crosstalkItr->second[5].xtalk_slope_left << " " 
     << crosstalkItr->second[5].xtalk_slope_right << " " 
     << crosstalkItr->second[5].xtalk_intercept_left << " " 
     << crosstalkItr->second[5].xtalk_intercept_right << std::endl;
  }
}


float CSCDbStripConditions::gain(const CSCDetId & detId, int channel) const
{
  assert(theGains != 0);
  int index = dbIndex(detId, channel);
  std::map< int,std::vector<CSCGains::Item> >::const_iterator layerGainsItr
    = theGains->gains.find(index);
  if(layerGainsItr == theGains->gains.end())
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find gain for layer " << detId;
  }

  return layerGainsItr->second[channel-1].gain_slope;
}


float CSCDbStripConditions::pedestal(const CSCDetId & detId, int channel) const
{
  assert(thePedestals != 0);
  int index = dbIndex(detId, channel);
  std::map< int,std::vector<CSCPedestals::Item> >::const_iterator pedestalItr
    = thePedestals->pedestals.find(index);
  if(pedestalItr == thePedestals->pedestals.end())
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find noise matrix for layer " << detId;
  }

  return pedestalItr->second[channel-1].ped;
}


float CSCDbStripConditions::pedestalVariance(const CSCDetId&detId, int channel) const
{
  assert(thePedestals != 0);
  int index = dbIndex(detId, channel);
  std::map< int,std::vector<CSCPedestals::Item> >::const_iterator pedestalItr
    = thePedestals->pedestals.find(index);
  if(pedestalItr == thePedestals->pedestals.end())
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find noise matrix for layer " << detId;
  }

  return pedestalItr->second[channel-1].rms;
}


void CSCDbStripConditions::crosstalk(const CSCDetId&detId, int channel,
                 double stripLength, bool leftRight,
                 float & capacitive, float & resistive) const
{
  assert(theCrosstalk != 0);
  int index = dbIndex(detId, channel);
  std::map< int,std::vector<CSCcrosstalk::Item> >::const_iterator crosstalkItr
    = theCrosstalk->crosstalk.find(index);
  if(crosstalkItr == theCrosstalk->crosstalk.end())
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find crosstalk for layer " << detId;
  }

  const CSCcrosstalk::Item & item = crosstalkItr->second[channel-1];
  float fraction = leftRight ? item.xtalk_intercept_right : item.xtalk_intercept_left;
  
  capacitive = theCapacitiveCrosstalk * fraction;
  resistive  = theResistiveCrosstalk;
}



void CSCDbStripConditions::fetchNoisifier(const CSCDetId & detId, int istrip)
{
  assert(theNoiseMatrix != 0);

  int index = dbIndex(detId, istrip);
  std::map< int,std::vector<CSCNoiseMatrix::Item> >::const_iterator matrixItr
    = theNoiseMatrix->matrix.find(index);
  if(matrixItr == theNoiseMatrix->matrix.end())
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find noise matrix for layer " << detId;
  }

  assert(matrixItr->second.size() < istrip);

  const CSCNoiseMatrix::Item & item = matrixItr->second[istrip-1];

  HepSymMatrix matrix(8);
  //TODO get the pedestals right
  matrix[3][3] = item.elem33;
  matrix[3][4] = item.elem34;
  matrix[3][5] = item.elem35;
  matrix[4][4] = item.elem44;
  matrix[4][5] = item.elem45;
  matrix[4][6] = item.elem46;
  matrix[5][5] = item.elem55;
  matrix[5][6] = item.elem56;
  matrix[5][7] = item.elem57;
  matrix[6][6] = item.elem66;
  matrix[6][7] = item.elem67;
  matrix[7][7] = item.elem77;
  
  if(theNoisifier != 0) delete theNoisifier;
  theNoisifier = new CorrelatedNoisifier(matrix);
}


int CSCDbStripConditions::dbIndex(const CSCDetId & id, int & channel)
{
  int ec = id.endcap();
  int st = id.station();
  int rg = id.ring();
  int ch = id.chamber();
  int la = id.layer();

  // there isn't really an ME1A.  It's channels 65-80 of ME11.
  if(st == 1 && rg == 4)
  {
    channel += 64;
    rg = 1;
  }

  return 220000000 + ec*100000 + st*10000 + rg*1000 + ch*10 + la;
}



