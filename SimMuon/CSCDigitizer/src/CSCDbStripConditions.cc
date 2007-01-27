#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
/*
    // Strip gains
    edm::ESHandle<CSCGains> hGains;
    theSetup->get<CSCGainsRcd>().get( hGains );
    const CSCGains* pGains = &*hGains.product();
    // Strip X-talk
    edm::ESHandle<CSCcrosstalk> hCrosstalk;
    theSetup->get<CSCcrosstalkRcd>().get( hCrosstalk );
    const CSCcrosstalk* pCrosstalk = &*hCrosstalk.product();
    // Strip autocorrelation noise matrix
    edm::ESHandle<CSCNoiseMatrix> hNoiseMatrix;
    theSetup->get<CSCNoiseMatrixRcd>().get(hNoiseMatrix);
    const CSCNoiseMatrix* pNoiseMatrix = &*hNoiseMatrix.product();

}

*/

void CSCDbStripConditions::fetchNoisifier(const CSCDetId & detId, int istrip)
{
  assert(theNoiseMatrix != 0);

  int index = dbIndex(detId);
  std::map< int,std::vector<CSCNoiseMatrix::Item> >::const_iterator layerMatrixItr
    = theNoiseMatrix->matrix.find(index);
  if(layerMatrixItr == theNoiseMatrix->matrix.end())
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find noise matrix for layer " << detId;
  }

  if(layerMatrixItr->second.size() < istrip)
  {
    throw cms::Exception("CSCDbStripConditions")
     << "Cannot find strip " << istrip
     << " in conditions  of size " << layerMatrixItr->second.size();
  }

  const CSCNoiseMatrix::Item & item = layerMatrixItr->second[istrip-1];

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


int CSCDbStripConditions::dbIndex(const CSCDetId & id)
{
  int ec = id.endcap();
  int st = id.station();
  int rg = id.ring();
  int ch = id.chamber();
  int la = id.layer();
  return 220000000 + ec*100000 + st*10000 + rg*1000 + ch*10 + la;
}



