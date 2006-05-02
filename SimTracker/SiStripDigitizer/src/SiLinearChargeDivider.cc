#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

SiLinearChargeDivider::SiLinearChargeDivider(const edm::ParameterSet& conf):conf_(conf){
  // Run APV in peak instead of deconvolution mode, which degrades the 
  // time resolution.
  //SimpleConfigurable<bool> SiLinearChargeDivider::peakMode(false,"SiStripDigitizer:APVpeakmode");
  peakMode=conf_.getParameter<bool>("APVpeakmode");
  
  // Enable interstrip Landau fluctuations within a cluster.
  //SimpleConfigurable<bool> SiLinearChargeDivider::fluctuateCharge(true,"SiStripDigitizer:LandauFluctuations");
  fluctuateCharge=conf_.getParameter<bool>("LandauFluctuations");
  
  // Number of segments per strip into which charge is divided during
  // simulation. If large, precision of simulation improves.
  //SimpleConfigurable<int> SiLinearChargeDivider::chargeDivisionsPerStrip(10,"SiStripDigitizer:chargeDivisionsPerStrip");
  chargedivisionsPerStrip=conf_.getParameter<int>("chargeDivisionsPerStrip");
 
  // delta cutoff in MeV, has to be same as in OSCAR (0.120425 MeV corresponding // to 100um range for electrons)
  //SimpleConfigurable<double>  SiLinearChargeDivider::deltaCut(0.120425,
  deltaCut=conf_.getParameter<double>("DeltaProductionCut");
}

SiChargeDivider::ionization_type 
SiLinearChargeDivider::divide(
			      const PSimHit& hit, const StripGeomDetUnit& det) {
 
  LocalVector direction = hit.exitPoint() - hit.entryPoint();  
  double pitch=100; // temporary!
  int NumberOfSegmentation =  
  //    (int)(1+chargeDivisionsPerStrip*fabs(direction.x())/det.specificTopology().pitch()); 
    (int)(1+chargedivisionsPerStrip*fabs(direction.x())/pitch); 
 
  float eLoss = hit.energyLoss();  // Eloss in GeV
 
  float decSignal = TimeResponse(hit);
 
  ionization_type _ionization_points;

  _ionization_points.resize(NumberOfSegmentation);

  //  cout<<"Modified SiStripDigitizer"<<endl;
  float energy;

  // Fluctuate charge in track subsegments
  float* eLossVector = new float[NumberOfSegmentation];
 
  if( fluctuateCharge ) {
    int pid = hit.particleType();
    float momentum = hit.pabs();
    float length = direction.mag();  // Track length in Silicon
    fluctuateEloss(pid, momentum, eLoss, length, NumberOfSegmentation, eLossVector);   
  }
 
  for ( int i = 0; i != NumberOfSegmentation; i++) {
    if( fluctuateCharge ) {
      energy=eLossVector[i]*decSignal/eLoss;
      EnergyDepositUnit edu(energy,hit.entryPoint()+float((i+0.5)/NumberOfSegmentation)*direction);//take energy value from vector eLossVector  
      _ionization_points[i] = edu; //save
    }else{
      energy=decSignal/float(NumberOfSegmentation);
      EnergyDepositUnit edu(energy,hit.entryPoint()+float((i+0.5)/NumberOfSegmentation)*direction);//take energy value from eLoss average over n.segments 
      _ionization_points[i] = edu; //save
    }
  }
 
  delete[] eLossVector;
  return _ionization_points;
}
    
void SiLinearChargeDivider::fluctuateEloss(int pid, float particleMomentum, 
				      float eloss, float length, 
				      int NumberOfSegs,float elossVector[]) {

  // Get dedx for this track
  float dedx;
  if( length > 0.) dedx = eloss/length;
  else dedx = eloss;

  double particleMass = 139.57; // Mass in MeV, Assume pion
  /*
  if( particleTable->getParticleData(pid) ) {  // Get mass from the PDTable
    particleMass = 1000. * particleTable->getParticleData(pid)->mass(); //Conv. GeV to MeV
  }
  */

  float segmentLength = length/NumberOfSegs;

  // Generate charge fluctuations.
  float de=0.;
  float sum=0.;
  double segmentEloss = (1000.*eloss)/NumberOfSegs; //eloss in MeV
  for (int i=0;i<NumberOfSegs;i++) {
    // The G4 routine needs momentum in MeV, mass in Mev, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV 
    // Returns fluctuated eloss in MeV
    //    double deltaCutoff = deltaCut.value(); // the cutoff is sometimes redefined inside, so fix it.
    double deltaCutoff = deltaCut;
    de = fluctuate.SampleFluctuations(double(particleMomentum*1000.),
				      particleMass, deltaCutoff, 
				      double(segmentLength*10.),
				      segmentEloss )/1000.; //convert to GeV
    elossVector[i]=de;
    sum +=de;
  }

  if(sum>0.) {  // If fluctuations give eloss>0.
    // Rescale to the same total eloss
    float ratio = eloss/sum;
    for (int ii=0;ii<NumberOfSegs;ii++) elossVector[ii]= ratio*elossVector[ii];
  } else {  // If fluctuations gives 0 eloss
    float averageEloss = eloss/NumberOfSegs;
    for (int ii=0;ii<NumberOfSegs;ii++) elossVector[ii]= averageEloss; 
  }
  return;
}

float SiLinearChargeDivider::TimeResponse( const PSimHit& hit ) {
  if (peakMode) {
    return this->PeakShape( hit );
  } else {
    return this->DeconvolutionShape( hit );
  }
}

float SiLinearChargeDivider::PeakShape(const PSimHit& hit){
  //  float dist = hit.det().position().mag();
  float dist = hit.localPosition().mag();
  float t0 = dist/30.;  // light velocity = 30 cm/ns
  float SigmaShape = 52.17; // From fit made by I.Tomalin to APV25 data presented by M.Raymond at LEB2000 conference.
  float tofNorm = (hit.tof() - t0)/SigmaShape;
  // Time when read out relative to time hit produced.
  float readTimeNorm = -tofNorm;
  // return the energyLoss weighted CR-RC shape peaked at t0.
  if (1 + readTimeNorm > 0) {
    return hit.energyLoss()*(1 + readTimeNorm)*exp(-readTimeNorm);
  } else {
    return 0.;
  }
}

float SiLinearChargeDivider::DeconvolutionShape(const PSimHit& hit){
  //  float dist = hit.det().position().mag();
  float dist = hit.localPosition().mag();
  float t0 = dist/30.;  // light velocity = 30 cm/ns
  float SigmaShape = 12.06; // From fit made by I.Tomalin to APV25 data presented by M.Raymond at LEB2000 conference.
  float tofNorm = (hit.tof() - t0)/SigmaShape;
  // Time when read out relative to time hit produced.
  float readTimeNorm = -tofNorm;
  // return the energyLoss weighted with a gaussian centered at t0 
  return hit.energyLoss()*exp(-0.5*readTimeNorm*readTimeNorm);
}

