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

  ileakParam_ = {0.993,-42.668}; //pedro 600V
//  ileakParam_ = {1.004,-43.059}; //fede 600V
//  ileakParam_ = {0.996,-42.464}; //fede 800V

  cellCapacitance_[HGCSiliconDetId::waferType::HGCalFine]=50;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThin]=65;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThick]=45;

  cellVolume_[HGCSiliconDetId::waferType::HGCalFine]=0.52*(120.e-4);
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThin]=1.18*(200.e-4);
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThick]=1.18*(300.e-4);

//  //  previous broken line
//  cceParam_[HGCSiliconDetId::waferType::HGCalFine]={1.5e+15, 6e+15, 1.33333e-17, -1.16778e-16, -2.58303e-17};          //120
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]={2e+15, 1e+17, -2.32205e-16, -5.7526e-17, -3.95409e-08};      //200
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick]={2.1e+14, 7.5e+14, -4.80627e-16, -1.00942e-15, -3.01253e-17};//300

//  line+log tdr 600V
cceParam_[HGCSiliconDetId::waferType::HGCalFine]        = {1.5e+15, -3.00394e-17, 0.318083};      //120
cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]  = {1.5e+15, -3.09878e-16, 0.211207};      //200
cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick] = {6e+14,   -7.96539e-16, 0.251751};      //300

//  //  line+log tdr 800V
//  cceParam_[HGCSiliconDetId::waferType::HGCalFine]        = {4.2e+15, 2.35482e-18, 0.553187};      //120
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]  = {1.5e+15, -1.98109e-16, 0.280567};      //200
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick] = {6e+14,   -5.24999e-16, 0.357616};      //300

//  //  line+log ttu 600V
//  cceParam_[HGCSiliconDetId::waferType::HGCalFine]        = {1.5e+15,  9.98631e-18, 0.343774};      //120
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]  = {1.5e+15, -2.17083e-16, 0.304873};      //200
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick] = {6e+14,   -8.01557e-16, 0.157375};      //300

//  //  line+log ttu 800V
//  cceParam_[HGCSiliconDetId::waferType::HGCalFine]        = {1.5e+15, 3.35246e-17, 0.251679};      //120
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]  = {1.5e+15, -1.62096e-16, 0.293828};      //200
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick] = {6e+14,   -5.95259e-16, 0.183929};      //300

//  //  line+log tdr 600V EPI
//  cceParam_[HGCSiliconDetId::waferType::HGCalFine]        = {3.5e+15, -9.73872e-19, 0.263812};      //100
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]  = {1.5e+15, -3.09878e-16, 0.211207};      //200
//  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick] = {6e+14,   -7.96539e-16, 0.251751};      //300

}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(SignalRange_t srange,const HGCSiliconDetId &cellId,bool ignoreFluence) {

  //compute the radius here
  GlobalPoint pt(geom()->getPosition(cellId));
  double radius(pt.perp());

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
    std::array<double, 8> radii{ {radius,pow(radius,2),pow(radius,3),pow(radius,4),0.,0.,0.,0.} };
    siop.fluence=getFluenceValue(cellId.subdet(),layer,radii);
    siop.lnfluence=log(siop.fluence);
    siop.ileak=exp(ileakParam_[0]*siop.lnfluence+ileakParam_[1])*cellVol*1e6;

    //lin+log parametrization
    siop.cce=siop.fluence<=cceParam_[cellThick][0] ? 1+cceParam_[cellThick][1]*siop.fluence :
      (1 - cceParam_[cellThick][2]*log(siop.fluence)) + (cceParam_[cellThick][1]*cceParam_[cellThick][0] + cceParam_[cellThick][2]*log(cceParam_[cellThick][0]));

    //previous broken line parametrization
    // siop.cce=siop.fluence<=cceParam_[cellThick][0] ? 1+cceParam_[cellThick][2]*siop.fluence :
    //   siop.fluence>cceParam_[cellThick][0] && siop.fluence<=cceParam_[cellThick][1] ? cceParam_[cellThick][3]*siop.fluence+(cceParam_[cellThick][2]-cceParam_[cellThick][3])*cceParam_[cellThick][0]+1 :
    //   cceParam_[cellThick][4]*siop.fluence+(cceParam_[cellThick][3]-cceParam_[cellThick][4])*cceParam_[cellThick][1]+(cceParam_[cellThick][2]-cceParam_[cellThick][3])*cceParam_[cellThick][0]+1;
  }


  //build noise estimate
  double enc_s(encsParam_[srange][0]+encsParam_[srange][1]*cellCap+encsParam_[srange][2]*pow(cellCap,2));
  double enc_p(encpScale_*sqrt(siop.ileak));
  siop.noise=hypot(enc_p,enc_s)*encCommonNoiseSub_*enc2fc_;

  return siop;
}
