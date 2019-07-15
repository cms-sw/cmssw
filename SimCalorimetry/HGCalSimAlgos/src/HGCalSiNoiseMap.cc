#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"

//
HGCalSiNoiseMap::HGCalSiNoiseMap() :
  encpScale_(840.),
  encCommonNoiseSub_( sqrt(1.25) ),
  enc2fc_(1.60217646E-4)
{
  encsParam_[q80fC]  = {636.,  15.6, 0.0328};
  encsParam_[q160fC] = {1045., 8.74, 0.0685};
  encsParam_[q320fC] = {1915., 2.79, 0.0878};

  cellCapacitance_[HGCSiliconDetId::waferType::HGCalFine]=50;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThin]=65;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThick]=45;

  cellVolume_[HGCSiliconDetId::waferType::HGCalFine]=0.52*(120.e-4);
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThin]=1.18*(200.e-4);
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThick]=1.18*(300.e-4);
}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(SignalRange_t srange,const HGCSiliconDetId &cellId,bool ignoreFluence) {

  SiCellOpCharacteristics siop;

  //decode cell properties
  int layer(cellId.layer());
  if(cellId.subdet()==DetId::HGCalEE) layer=(layer-1)/2+1;
  HGCSiliconDetId::waferType cellThick( HGCSiliconDetId::waferType(cellId.type()) );
  double cellCap(cellCapacitance_[cellThick]);
  double cellVol(cellVolume_[cellThick]);

  //get fluence
  if(getDoseMap().empty()) return siop;

  //leakage current and CCE [muA]
  if(ignoreFluence) {
    siop.fluence=0;
    siop.lnfluence=-1;
    siop.ileak=exp(ileakParam_[1])*cellVol*1e6;
    siop.cce=1;
  }
  else {
    //compute the radius here
    auto xy(ddd()->locateCell(cellId.layer(), cellId.waferU(), cellId.waferV(), cellId.cellU(), cellId.cellV(), true, true));
    double radius2 = std::pow(xy.first, 2) + std::pow(xy.second, 2); //in cm

    double radius  = sqrt(radius2);
    double radius3 = radius*radius2;
    double radius4 = pow(radius2,2);
    std::array<double, 8> radii{ {radius,radius2,radius3,radius4,0.,0.,0.,0.} };
    siop.fluence=getFluenceValue(cellId.subdet(),layer,radii);
    siop.lnfluence=log(siop.fluence);
    siop.ileak=exp(ileakParam_[0]*siop.lnfluence+ileakParam_[1])*cellVol*1e6;

    //lin+log parametrization
    siop.cce=siop.fluence<=cceParam_[cellThick][0] ? 1+cceParam_[cellThick][1]*siop.fluence :
      (1 - cceParam_[cellThick][2]*log(siop.fluence)) + (cceParam_[cellThick][1]*cceParam_[cellThick][0] + cceParam_[cellThick][2]*log(cceParam_[cellThick][0]));
  }


  //build noise estimate
  double enc_s(encsParam_[srange][0]+encsParam_[srange][1]*cellCap+encsParam_[srange][2]*pow(cellCap,2));
  double enc_p(encpScale_*sqrt(siop.ileak));
  siop.noise=hypot(enc_p,enc_s)*encCommonNoiseSub_*enc2fc_;

  return siop;
}
