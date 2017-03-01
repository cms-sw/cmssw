#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModel.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

void 
ME0DigiPreRecoModel::fillDigis(int rollDetId, ME0DigiPreRecoCollection& digis)
{
  for (auto d: digi_)
  {
    digis.insertDigi(ME0DetId(rollDetId), d);
  }
  digi_.clear();
}

