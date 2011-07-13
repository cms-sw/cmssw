///////////////////////////////////////////////////////////////////////////////
// File: ChargeDrifterFP420
// Date: 08.2008
// Description: ChargeDrifterFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/ChargeDrifterFP420.h"
using namespace std;

ChargeDrifterFP420::ChargeDrifterFP420(                              double mt,
								     double tn,
								     double dc,
								     double tm,
								     double cdr,
								     double dv,
								     double av,
								     double ptx,
								     double pty, int verbosity) {
  //
  //
  verbo=verbosity; 
  modulePath = mt;
  constTe = tn;
  constDe = dc;
  temperature = tm;// keep just in case
  startT0 = cdr;
  depV = dv;
  appV = av;
  
  ldriftcurrX = ptx;
  ldriftcurrY = pty;
  //
  //
  
  
  
  //      edm::LogInfo("ChargeDrifterFP420") << "call constructor";
  if(verbo>0) {
    std::cout << "ChargeDrifterFP420: call constructor" << std::endl;
    std::cout << "ldriftcurrX= " << ldriftcurrX << "ldriftcurrY= " << ldriftcurrY << std::endl;
    std::cout << "modulePath= " << modulePath << "constTe= " << constTe << std::endl;
    std::cout << "ChargeDrifterFP420:----------------------" << std::endl;
  }
}



CDrifterFP420::collection_type ChargeDrifterFP420::drift(const CDrifterFP420::ionization_type ion, const G4ThreeVector& driftDir,const int& xytype){
  //
  //
  if(verbo>0) {
    std::cout << "ChargeDrifterFP420: collection_type call drift" << std::endl;
  }
  //
  //
  collection_type _temp;
  _temp.resize(ion.size());
  
  for (unsigned int i=0; i<ion.size(); i++){
    _temp[i] = drift(ion[i], driftDir, xytype);
  }
  return _temp;
}

AmplitudeSegmentFP420 ChargeDrifterFP420::drift
(const EnergySegmentFP420& edu, const G4ThreeVector& drift, const int& xytype){
  //
  //
  //    sliX sliY sliZ are LOCALl(!) coordinates coming from ...hit.getEntryLocalP()...  
  //
  //
  //                Exchange xytype 1 <-> 2
  double sliX = (edu).x();
  double sliY = (edu).y();
  double sliZ = (edu).z();
  
  double currX = sliX;
  double currY = sliY;
  double currZ = sliZ;
  
  double pathFraction=0., pathValue=0.;
  double tanShiftAngleX=0.,tanShiftAngleY=0.,tanShiftAngleZ=0.;
  double xDriftDueToField=0.,yDriftDueToField=0.,zDriftDueToField=0.;
  if(verbo>0) {
    std::cout << "=================================================================== " << std::endl;
    std::cout << "ChargeDrifterFP420: xytype= " << xytype << std::endl;
    std::cout << "constTe= " <<  constTe << "ldriftcurrX= " <<  ldriftcurrX << "ldriftcurrY= " <<  ldriftcurrY << std::endl;
    std::cout << "1currX = " << currX  << "  1currY = " << currY  << "  1currZ = " << currZ  << std::endl;
    std::cout << "drift.x() = " << drift.x()  << "  drift.y() = " << drift.y()  << "  drift.z() = " << drift.z()  << std::endl;
  }
  
  // Yglobal Xlocal:
  //
  //
  // will define drift time of electrons along Xlocal or Ygobal with ldriftcurrY from Yglobal
  // change of ldriftcurrX to ldriftcurrY is done in intialization of ChargeDrifterFP420, so use old names for ldriftcurres
  if(xytype == 2) {
    tanShiftAngleY = drift.y()/drift.x();
    tanShiftAngleZ = drift.z()/drift.x();
    
    //    pathValue = fabs(sliX-ldriftcurrX*int(sliX/ldriftcurrX));
    pathValue = fabs(sliX)-int(fabs(sliX/(2*ldriftcurrX))+0.5)*(2*ldriftcurrX);
    pathValue = fabs(pathValue);
    pathFraction = pathValue/ldriftcurrX; 
    if(verbo>0) {
      std::cout << "==================================" << std::endl;
      std::cout << "fabs(sliX)= " << fabs(sliX) << "ldriftcurrX= " << ldriftcurrX << std::endl;
      std::cout << "fabs(sliX/(2*ldriftcurrX))+0.5= " << fabs(sliX/(2*ldriftcurrX))+0.5 << std::endl;
      std::cout << "int(fabs(sliX/(2*ldriftcurrX))+0.5)*(2*ldriftcurrX)= " << int(fabs(sliX/(2*ldriftcurrX))+0.5)*(2*ldriftcurrX) << std::endl;
      std::cout << "pathValue= " << pathValue << std::endl;
      std::cout << "pathFraction= " << pathFraction << std::endl;
      std::cout << "==================================" << std::endl;
    }
    
    pathFraction = pathFraction>0. ? pathFraction : 0. ;
    pathFraction = pathFraction<1. ? pathFraction : 1. ;
    
    yDriftDueToField // Drift along Y due to BField
      = pathValue*tanShiftAngleY;
    zDriftDueToField // Drift along Z due to BField
      = pathValue*tanShiftAngleZ;
    // will define Y and Z ccordinates (E along X)
    currY = sliY + yDriftDueToField;
    currZ = sliZ + zDriftDueToField;
  }
  
  // Xglobal Ylocal:
  // will define drift time of electrons along Ylocal
  else if(xytype == 1) {
    tanShiftAngleX = drift.x()/drift.y();
    tanShiftAngleZ = drift.z()/drift.y();
    
    //pathValue = fabs(sliY-ldriftcurrY*int(sliY/ldriftcurrY));
    pathValue = fabs(sliY)-int(fabs(sliY/(2*ldriftcurrY))+0.5)*(2*ldriftcurrY);
    pathValue = fabs(pathValue);
    pathFraction = pathValue/ldriftcurrY; 
    //
    //
    
    if(verbo>0) {
      std::cout << "==================================" << std::endl;
      std::cout << "fabs(sliY)= " << fabs(sliY) << "ldriftcurrY= " << ldriftcurrY << std::endl;
      std::cout << "fabs(sliY/(2*ldriftcurrY))+0.5= " << fabs(sliY/(2*ldriftcurrY))+0.5 << std::endl;
      std::cout << "int(fabs(sliY/(2*ldriftcurrY))+0.5)*(2*ldriftcurrY)= " << int(fabs(sliY/(2*ldriftcurrY))+0.5)*(2*ldriftcurrY) << std::endl;
      std::cout << "pathValue= " << pathValue << std::endl;
      std::cout << "pathFraction= " << pathFraction << std::endl;
      std::cout << "==================================" << std::endl;
    }
    
    if(pathFraction<0. || pathFraction>1.) std::cout << "ChargeDrifterFP420: ERROR:pathFraction=" << pathFraction << std::endl;
    pathFraction = pathFraction>0. ? pathFraction : 0. ;
    pathFraction = pathFraction<1. ? pathFraction : 1. ;
    xDriftDueToField // Drift along X due to BField
      = pathValue*tanShiftAngleX;
    //      = (ldriftcurrY-sliY)*tanShiftAngleX;
    zDriftDueToField // Drift along Z due to BField
      = pathValue*tanShiftAngleZ;
    // will define X and Z ccordinates (E along Y)
    currX = sliX + xDriftDueToField;
    currZ = sliZ + zDriftDueToField;
  }
  //  double tanShiftAngleX = drift.x()/drift.z();
  //  double tanShiftAngleY = drift.y()/drift.z();
  // double pathFraction = (modulePath/2.-sliZ)/modulePath ; 
  // pathFraction = pathFraction>0. ? pathFraction : 0. ;
  // pathFraction = pathFraction<1. ? pathFraction : 1. ;
  
  //uble xDriftDueToField // Drift along X due to BField
  //= (modulePath/2. - sliZ)*tanShiftAngleX;
  //uble yDriftDueToField // Drift along Y due to BField
  //= (modulePath/2. - sliZ)*tanShiftAngleY;
  //uble currX = sliX + xDriftDueToField;
  //uble currY = sliY + yDriftDueToField;
  //  
  //  
  // log is a ln 
  //  std::cout << "ChargeDrifterFP420: depV=" <<depV  << "  appV=" << appV << " startT0 =" <<startT0  << " 1.-2*depV*pathFraction/(depV+appV) =" <<1.-2*depV*pathFraction/(depV+appV)  << " log(1.-2*depV*pathFraction/(depV+appV)) =" <<log(1.-2*depV*pathFraction/(depV+appV))  << std::endl;
  double bbb = 1.-2*depV*pathFraction/(depV+appV);
  if(bbb<0.) std::cout << "ChargeDrifterFP420:ERROR: check your Voltage for log(bbb) bbb=" << bbb << std::endl;
  double driftTime = -constTe*log(bbb) + startT0;  
  //    log(1.-2*depV*pathFraction/(depV+appV)) + startT0;  
  // since no magnetic field the Sigma_x, Sigma_y are the same =  sigma  !!!!!!!!!!!!!!!!! 
  double sigma = sqrt(2.*constDe*driftTime*100.);  //  * 100.  - since constDe is [cm2/sec], but i want [mm2/sec]
  
  //  std::cout << "ChargeDrifterFP420: driftTime=  " << driftTime << "  pathFraction=  " << pathFraction << "  constTe=  " << constTe << "  sigma=  " << sigma << std::endl;
  if(verbo>0) {
    std::cout << "ChargeDrifterFP420: drift: xytype=" << xytype << "pathFraction=" << pathFraction << std::endl;
    std::cout << "  constTe= " << constTe  << "  driftTime = " << driftTime << " startT0  = " << startT0 <<  std::endl;
    std::cout << " log = " << log(1.-2*depV*pathFraction/(depV+appV))  << std::endl;
    std::cout << " negativ inside log = " << -2*depV*pathFraction/(depV+appV)  << std::endl;
    std::cout << " constDe = " << constDe  << "  sigma = " << sigma << std::endl;
    
    std::cout << "ChargeDrifterFP420: drift: xytype=" << xytype << "pathValue=" << pathValue << std::endl;
    std::cout << " tanShiftAngleX = " << tanShiftAngleX  << "  tanShiftAngleY = " << tanShiftAngleY  << "  tanShiftAngleZ = " << tanShiftAngleZ  << std::endl;
    std::cout << "sliX = " << sliX  << "  sliY = " << sliY  << "  sliZ = " << sliZ  << std::endl;
    std::cout << "pathFraction = " << pathFraction  << "  driftTime = " << driftTime  << std::endl;
    std::cout << "sigma = " << sigma  << std::endl;
    std::cout << "xDriftDueToField = " << xDriftDueToField  << "  yDriftDueToField = " << yDriftDueToField  << "  zDriftDueToField = " << zDriftDueToField  << std::endl;
    std::cout << "2currX = " << currX  << "  2currY = " << currY  << "  2currZ = " << currZ  << std::endl;
    std::cout << "ChargeDrifterFP420: drift; finally, rETURN AmplitudeSlimentFP420" << std::endl;
    std::cout << "===================================================================" << std::endl;
    std::cout << " (edu).energy()= " << (edu).energy() << std::endl;
    std::cout << "==" << std::endl;
  }
  //  std::cout << "ChargeDrifterFP420: (edu).energy()= " << (edu).energy() << std::endl;
  return AmplitudeSegmentFP420(currX,currY,currZ,sigma,
			       (edu).energy());  
}

