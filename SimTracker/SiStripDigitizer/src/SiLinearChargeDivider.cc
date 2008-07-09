#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiLinearChargeDivider::SiLinearChargeDivider(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng):
  conf_(conf),rndEngine(eng),theParticleDataTable(0){
  // Run APV in peak instead of deconvolution mode, which degrades the time resolution.
  peakMode=conf_.getParameter<bool>("APVpeakmode");
  
  // APV time resolution
  timeResPeak=conf_.getParameter<double>("SigmaShapePeak");
  timeResDeco=conf_.getParameter<double>("SigmaShapeDeco");

  // Enable interstrip Landau fluctuations within a cluster.
  fluctuateCharge=conf_.getParameter<bool>("LandauFluctuations");
  
  // Number of segments per strip into which charge is divided during
  // simulation. If large, precision of simulation improves.
  chargedivisionsPerStrip=conf_.getParameter<int>("chargeDivisionsPerStrip");
 
  // delta cutoff in MeV, has to be same as in OSCAR (0.120425 MeV corresponding // to 100um range for electrons)
  deltaCut=conf_.getParameter<double>("DeltaProductionCut");

  //Offset for digitization during the MTCC and in general for taking cosmic particle
  //The value to be used it must be evaluated and depend on the volume defnition used
  //for the cosimc generation (Considering only the tracker the value is 11 ns)
  cosmicShift=conf_.getUntrackedParameter<double>("CosmicDelayShift");
  
  fluctuate = new SiG4UniversalFluctuation(rndEngine);
}

SiLinearChargeDivider::~SiLinearChargeDivider(){
  delete fluctuate;
}

SiChargeDivider::ionization_type 
SiLinearChargeDivider::divide(const PSimHit& hit, const LocalVector& driftdir, double moduleThickness, const StripGeomDetUnit& det) {

  int NumberOfSegmentation =  
    (int)(1+chargedivisionsPerStrip*fabs(driftXPos(hit.exitPoint(), driftdir, moduleThickness)-driftXPos(hit.entryPoint(), driftdir, moduleThickness))/(det.specificTopology()).localPitch(hit.localPosition())); 
 
  float eLoss = hit.energyLoss();  // Eloss in GeV
 
  float decSignal = TimeResponse(hit, det);
 
  ionization_type _ionization_points;

  _ionization_points.resize(NumberOfSegmentation);

  float energy;

  // Fluctuate charge in track subsegments

  LocalVector direction = hit.exitPoint() - hit.entryPoint();  

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

  assert(theParticleDataTable != 0);
  ParticleData const * particle = theParticleDataTable->particle( pid );
  double particleMass = 139.57;              // Mass in MeV, Assume pion
  if(particle == 0)
    {
      LogDebug("SiLinearChargeDivider") << "Cannot find particle of type "<<pid
					<< " in the PDT we assign to this particle the mass of the Pion";
    }
  else
    {
      particleMass = particle->mass()*1000; // Mass in MeV
    }

  //This is a temporary fix for protect from particles with Mass = 0
  if(fabs(particleMass)<1.e-6 || pid == 22)
    particleMass = 139.57;

  float segmentLength = length/NumberOfSegs;

  // Generate charge fluctuations.
  float de=0.;
  float sum=0.;
  double segmentEloss = (1000.*eloss)/NumberOfSegs; //eloss in MeV
  for (int i=0;i<NumberOfSegs;i++) {
    // The G4 routine needs momentum in MeV, mass in MeV, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV 
    // Returns fluctuated eloss in MeV
    // the cutoff is sometimes redefined inside, so fix it.
    double deltaCutoff = deltaCut;
    de = fluctuate->SampleFluctuations(double(particleMomentum*1000.),
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

float SiLinearChargeDivider::TimeResponse( const PSimHit& hit, const StripGeomDetUnit& det ) {
  if (peakMode) {
    return this->PeakShape( hit, det );
  } else {
    return this->DeconvolutionShape( hit, det );
  }
}

float SiLinearChargeDivider::PeakShape(const PSimHit& hit, const StripGeomDetUnit& det){
  float dist = det.surface().toGlobal(hit.localPosition()).mag();
  float t0 = dist/30.;  // light velocity = 30 cm/ns
  float SigmaShape = timeResPeak; // 52.17 ns from fit made by I.Tomalin to APV25 data presented by M.Raymond at LEB2000 conference.
  float tofNorm = (hit.tof() - cosmicShift - t0)/SigmaShape;
  // Time when read out relative to time hit produced.
  float readTimeNorm = -tofNorm;
  // return the energyLoss weighted CR-RC shape peaked at t0.
  if (1 + readTimeNorm > 0) {
    return hit.energyLoss()*(1 + readTimeNorm)*exp(-readTimeNorm);
  } else {
    return 0.;
  }
}

float SiLinearChargeDivider::DeconvolutionShape(const PSimHit& hit, const StripGeomDetUnit& det){
  float dist = det.surface().toGlobal(hit.localPosition()).mag();
  float t0 = dist/30.;  // light velocity = 30 cm/ns
  float SigmaShape = timeResDeco; // 12.06 ns from fit made by I.Tomalin to APV25 data presented by M.Raymond at LEB2000 conference.
  float tofNorm = (hit.tof() - cosmicShift - t0)/SigmaShape;
  // Time when read out relative to time hit produced.
  float readTimeNorm = -tofNorm;
  // return the energyLoss weighted with a gaussian centered at t0 
  return hit.energyLoss()*exp(-0.5*readTimeNorm*readTimeNorm);
}

void SiLinearChargeDivider::setParticleDataTable(const ParticleDataTable * pdt)
{
  theParticleDataTable = pdt;
}


float SiLinearChargeDivider::driftXPos
(const Local3DPoint& pos, const LocalVector& drift, double moduleThickness){
  
  double tanLorentzAngleX = drift.x()/drift.z();
  
  double segX = pos.x();
  double segZ = pos.z();
  
  double thicknessFraction = (moduleThickness/2.-segZ)/moduleThickness ; 
  // fix the bug due to  rounding on entry and exit point
  thicknessFraction = thicknessFraction>0. ? thicknessFraction : 0. ;
  thicknessFraction = thicknessFraction<1. ? thicknessFraction : 1. ;
  
  double xDriftDueToMagField // Drift along X due to BField
    = (moduleThickness/2. - segZ)*tanLorentzAngleX;
  double positionX = segX + xDriftDueToMagField;

  return positionX;
		     
}
