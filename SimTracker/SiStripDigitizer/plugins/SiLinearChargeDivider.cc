#include "SiLinearChargeDivider.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiLinearChargeDivider::SiLinearChargeDivider(const edm::ParameterSet& conf) :
  // Run APV in peak instead of deconvolution mode, which degrades the time resolution.
  peakMode(conf.getParameter<bool>("APVpeakmode")),
  // Enable interstrip Landau fluctuations within a cluster.
  fluctuateCharge(conf.getParameter<bool>("LandauFluctuations")),
  // Number of segments per strip into which charge is divided during
  // simulation. If large, precision of simulation improves.
  chargedivisionsPerStrip(conf.getParameter<int>("chargeDivisionsPerStrip")),
  // delta cutoff in MeV, has to be same as in Geant (0.120425 MeV corresponding to 100um range for electrons)
  deltaCut(conf.getParameter<double>("DeltaProductionCut")),
  //Offset for digitization during the MTCC and in general for taking cosmic particle
  //The value to be used it must be evaluated and depend on the volume defnition used
  //for the cosimc generation (Considering only the tracker the value is 11 ns)
  cosmicShift(conf.getUntrackedParameter<double>("CosmicDelayShift")),
  
  // Geant4 engine used to fluctuate the charge from segment to segment

  t0IdxPeak(conf.getParameter<edm::ParameterSet>("peak").getParameter<int>("t0Idx")),
  resolutionPeak(conf.getParameter<edm::ParameterSet>("peak").getParameter<double>("resolution")),
  valuesPeak(conf.getParameter<edm::ParameterSet>("peak").getParameter<std::vector<double> >("values")),

  t0IdxDeco(conf.getParameter<edm::ParameterSet>("deco").getParameter<int>("t0Idx")),
  resolutionDeco(conf.getParameter<edm::ParameterSet>("deco").getParameter<double>("resolution")),
  valuesDeco(conf.getParameter<edm::ParameterSet>("deco").getParameter<std::vector<double> > ("values")),
  
  theParticleDataTable(nullptr),
  fluctuate(new SiG4UniversalFluctuation())
  
{
  if (peakMode){
    if (valuesPeak.at(t0IdxPeak)!=1) edm::LogError("WrongAPVPulseShape")<<"The max value of the APV pulse shape stored in the valuesPeak vector in SimGeneral/MixingModule/python/SiStripSimParameters_cfi.py is not equal to 1. Need to be fixed."<<std::endl;
  }

  else{
    if (valuesDeco.at(t0IdxDeco)!=1) edm::LogError("WrongAPVPulseShape")<<"The max value of the APV pulse shape stored in the valuesDeco vector in SimGeneral/MixingModule/python/SiStripSimParameters_cfi.py is not equal to 1. Need to be fixed."<<std::endl;   
  }
}
    
SiLinearChargeDivider::~SiLinearChargeDivider(){
}

SiChargeDivider::ionization_type 
SiLinearChargeDivider::divide(const PSimHit* hit, const LocalVector& driftdir, double moduleThickness, const StripGeomDetUnit& det, CLHEP::HepRandomEngine* engine) {

  // signal after pulse shape correction
  float const decSignal = TimeResponse(hit, det);

  // if out of time go home!
  if (0==decSignal) return ionization_type();

  // Get the nass if the particle, in MeV.
  // Protect from particles with Mass = 0, assuming then the pion mass
  assert(theParticleDataTable != nullptr);
  ParticleData const * particle = theParticleDataTable->particle( hit->particleType() );
  double const particleMass = particle ? particle->mass()*1000 : 139.57;
  double const particleCharge = particle ? particle->charge() : 1.;

  if(!particle) {
    LogDebug("SiLinearChargeDivider") << "Cannot find particle of type "<< hit->particleType() 
                                      << " in the PDT we assign to this particle the mass and charge of the Pion";
  }

  int NumberOfSegmentation = 
  // if neutral: just one deposit....
    (fabs(particleMass)<1.e-6 || particleCharge==0) ? 1 :
    // computes the number of segments from number of segments per strip times number of strips.
    (int)(1 + chargedivisionsPerStrip*fabs(driftXPos(hit->exitPoint(), driftdir, moduleThickness)-
					   driftXPos(hit->entryPoint(), driftdir, moduleThickness) )
	  /det.specificTopology().localPitch(hit->localPosition())         ); 
 

  // Eloss in GeV
  float eLoss = hit->energyLoss();


  // Prepare output
  ionization_type _ionization_points;
  _ionization_points.resize(NumberOfSegmentation);

  // Fluctuate charge in track subsegments
  LocalVector direction = hit->exitPoint() - hit->entryPoint();
  if (NumberOfSegmentation <=1) {
    // here I need a random... not 0.5
    _ionization_points[0] = EnergyDepositUnit(eLoss*decSignal/eLoss,
					      hit->entryPoint()+0.5f*direction);
  }else {
    float eLossVector[NumberOfSegmentation];
    if( fluctuateCharge ) {
      fluctuateEloss(particleMass, hit->pabs(), eLoss, direction.mag(), NumberOfSegmentation, eLossVector, engine);
      // Save the energy of each segment
      for ( int i = 0; i != NumberOfSegmentation; i++) {
	// take energy value from vector eLossVector, 
	_ionization_points[i] = EnergyDepositUnit(eLossVector[i]*decSignal/eLoss,
						  hit->entryPoint()+float((i+0.5)/NumberOfSegmentation)*direction);
      }
    } else {
      // Save the energy of each segment
      for ( int i = 0; i != NumberOfSegmentation; i++) {
	// take energy value from eLoss average over n.segments.
	_ionization_points[i] = EnergyDepositUnit(decSignal/float(NumberOfSegmentation),
						  hit->entryPoint()+float((i+0.5)/NumberOfSegmentation)*direction);
      }
    }
  }
  return _ionization_points;
}

void SiLinearChargeDivider::fluctuateEloss(double particleMass, float particleMomentum, 
                                           float eloss, float length, 
                                           int NumberOfSegs,float elossVector[],
                                           CLHEP::HepRandomEngine* engine) {


  // Generate charge fluctuations.
  float sum=0.;
  double deltaCutoff;
  double mom = particleMomentum*1000.;
  double seglen = length/NumberOfSegs*10.;
  double segeloss = (1000.*eloss)/NumberOfSegs;
  for (int i=0;i<NumberOfSegs;i++) {
    // The G4 routine needs momentum in MeV, mass in MeV, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV 
    // Returns fluctuated eloss in MeV
    // the cutoff is sometimes redefined inside, so fix it.
    deltaCutoff = deltaCut;
    sum += (elossVector[i] = fluctuate->SampleFluctuations(mom,particleMass,deltaCutoff,seglen,segeloss, engine)/1000.);
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

 float SiLinearChargeDivider::PeakShape(const PSimHit* hit, const StripGeomDetUnit& det){
   // x is difference between the tof and the tof for a photon (reference)
   // converted into a bin number
   const auto dTOF = det.surface().toGlobal(hit->localPosition()).mag()/30. + cosmicShift - hit->tof();
   const int x= int(dTOF*resolutionPeak) + t0IdxPeak;
   if(x < 0 || x >= int(valuesPeak.size())) return 0;
   return hit->energyLoss()*valuesPeak[std::size_t(x)];
 }

 float SiLinearChargeDivider::DeconvolutionShape(const PSimHit* hit, const StripGeomDetUnit& det){
   // x is difference between the tof and the tof for a photon (reference)
   // converted into a bin number
   const auto dTOF = det.surface().toGlobal(hit->localPosition()).mag()/30. + cosmicShift - hit->tof();
   const int x= int(dTOF*resolutionDeco) + t0IdxDeco;
   if(x < 0 || x >= int(valuesDeco.size())) return 0;
   return hit->energyLoss()*valuesDeco[std::size_t(x)];
 }
