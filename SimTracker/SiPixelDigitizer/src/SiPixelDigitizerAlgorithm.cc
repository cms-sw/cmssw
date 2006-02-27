//class SiPixelDigitizerAlgorithm SimTracker/SiPixelDigitizer/src/SiPixelDigitizerAlgoithm.cc

// Original Author Danek Kotlinsky
// Ported in CMSSW by  Michele Pioppi-INFN perugia
//         Created:  Mon Sep 26 11:08:32 CEST 2005



#include <vector>
#include <iostream>
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"
#include <gsl/gsl_sf_erf.h>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
#include "SimTracker/SiPixelDigitizer/interface/PixelChipIndices.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;
using namespace edm;
SiPixelDigitizerAlgorithm::SiPixelDigitizerAlgorithm(const edm::ParameterSet& conf):conf_(conf){
  // Common pixel parameters
  // This are parameters which are not likely to be changed
  NumberOfSegments = 20; // Default number of track segment divisions
  ClusterWidth = 3.;     // Charge integration spread on the collection plane
  GeVperElectron = 3.7E-09;  // 1 electrons=3.7eV, 1keV=270.3e
  Sigma0 = 0.0007;           // Charge diffusion constant 
  Dist300 = 0.0300;          //   normalized to 300micron Silicon

  //get external parameters
  // ADC calibration 1adc count = 135e.
  // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc]=2[adc/kev]
  // Be carefull, this parameter is also used in SiPixelDet.cc to 
  // calculate the noise in adc counts from noise in electrons.
  // Both defaults should be the same.
  theElectronPerADC=conf_.getUntrackedParameter<double>("ElectronPerAdc",135.0);

  // ADC saturation value, 255=8bit adc.
  theAdcFullScale=conf_.getUntrackedParameter<int>("AdcFullScale",255);

  // Pixel threshold in units of noise.
  thePixelThreshold=conf_.getUntrackedParameter<double>("ThresholdInNoiseUnits",5.);

  // Noise in electrons.
  theNoiseInElectrons=conf_.getUntrackedParameter<double>("NoiseInElectrons",500.0);

  //theTofCut 12.5, cut in particle TOD +/- 12.5ns
  theTofCut=conf_.getUntrackedParameter<double>("TofCut",12.5);

  //Lorentz angle tangent per Tesla
  tanLorentzAnglePerTesla=conf_.getUntrackedParameter<double>("TanLorentzAnglePerTesla",0.106);

  // Add noise   
  addNoise=conf_.getUntrackedParameter<bool>("AddNoise",true);

  // Add noisy pixels 
  addNoisyPixels=conf_.getUntrackedParameter<bool>("AddNoisyPixels",true);

  // Fluctuate charge in track subsegments
  fluctuateCharge=conf_.getUntrackedParameter<bool>("FluctuateCharge",true);

  // delta cutoff in MeV, has to be same as in OSCAR=0.030/cmsim=1.0 MeV
  //tMax = 0.030; // In MeV.  
  tMax =conf_.getUntrackedParameter<double>("DeltaProductionCut",0.030);  
 
  thePixelLuminosity=conf_.getUntrackedParameter<int>("AddPixelInefficiency",1);
  // Get the constants for the miss-calibration studies
  doMissCalibrate=conf_.getUntrackedParameter<bool>("MissCalibrate",false); // Enable miss-calibration
  theGainSmearing=conf_.getUntrackedParameter<double>("GainSmearing",0.0); // sigma of the gain smearing
  theOffsetSmearing=conf_.getUntrackedParameter<double>("OffsetSmearing",0.0); //sigma of the offset smearing




  //pixel inefficiency
  
  if (thePixelLuminosity==0) {
    pixelInefficiency=false;
    for (int i=0; i<6;i++) {
      thePixelEfficiency[i]     = 1.;  // pixels = 100%
      // For columns make 1% default.
      thePixelColEfficiency[i]  = 1.;  // columns = 100%
      // A flat 0.25% inefficiency due to lost data packets from TBM
      thePixelChipEfficiency[i] = 1.; // chips = 100%
    }
  }
  
  if (thePixelLuminosity>0) {
    pixelInefficiency=true;
    // Default efficiencies 
    for (int i=0; i<6;i++) {
      // Assume 1% inefficiency for single pixels, 
      // this is given by faulty bump-bonding and seus.  
      thePixelEfficiency[i]     = 1.-0.01;  // pixels = 99%
      // For columns make 1% default.
      thePixelColEfficiency[i]  = 1.-0.01;  // columns = 99%
      // A flat 0.25% inefficiency due to lost data packets from TBM
      thePixelChipEfficiency[i] = 1.-0.0025; // chips = 99.75%
    }
    
    
    
    // Special cases 

    //   unsigned int Subid=DetId(detID).subdetId();
   
    if(thePixelLuminosity==10) { // For high luminosity
      thePixelColEfficiency[0] = 1.-0.034; // 3.4% for r=4 only
      thePixelEfficiency[0]    = 1.-0.015; // 1.5% for r=4
    }

  }



  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    LogDebug ("PixelDigitizer ") <<"SiPixelDigitizerAlgorithm constructed"
				 <<"Configuraion parameters:" 
				 << "Threshold/Gain = "  
				 << thePixelThreshold << " " <<  theElectronPerADC 
				 << " " << theAdcFullScale 
				 << " The delta cut-off is set to " << tMax;
    if(doMissCalibrate) LogDebug ("PixelDigitizer ") << " miss-calibrate the pixel amplitude " 
						     << theGainSmearing << " " << theOffsetSmearing ;
  }
  //MP DA RISOLVERE
  //   particleTable =  &HepPDT::theTable();

}
SiPixelDigitizerAlgorithm::~SiPixelDigitizerAlgorithm(){
 if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
   LogDebug ("PixelDigitizer")<<"SiPixelDigitizerAlgorithm deleted";
 }
}

vector<PixelDigi>  SiPixelDigitizerAlgorithm::run(const std::vector<PSimHit> &input,
						  PixelGeomDetUnit *pixdet,
						  GlobalVector bfield)
{


  _detp = pixdet; //cache the PixGeomDetUnit
  _PixelHits=input; //cache the SimHit
  _bfield=bfield; //cache the drift direction


  // Pixel Efficiency moved from the constructor to the method run because
  // the information of the det are nota available in the constructor
  // Effciency parameters. 0 - no inefficiency, 1-low lumi, 10-high lumi



  detID= _detp->geographicalId().rawId();






  _signal.clear();

  // initalization  of pixeldigisimlinks
  link_coll.clear();

 //Digitization of the SimHits of a given pixdet
  vector<PixelDigi> collector =digitize(pixdet);

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    LogDebug ("PixelDigitizer") << "[SiPixelDigitizerAlgorithm] converted " << collector.size() << " PixelDigis in DetUnit" << detID; 
   }
   return collector;
}
/**********************************************************************/

vector<PixelDigi> SiPixelDigitizerAlgorithm::digitize(PixelGeomDetUnit *det){
  


  if( _PixelHits.size() > 0 || addNoisyPixels) {
 
    topol=&det->specificTopology(); // cache topology
    numColumns = topol->ncolumns();  // det module number of cols&rows
    numRows = topol->nrows();


    moduleThickness = det->specificSurface().bounds().thickness(); // full detector thicness

    //MP DA SISTEMARE
    //     float noiseInADCCounts = _detp->readout().noiseInAdcCounts();
    //  float noiseInADCCounts=3.7;  
    // For the noise generation I need noise in electrons
    // theNoiseInElectrons = noiseInADCCounts * theElectronPerADC;    
    // Find the threshold in electrons
    thePixelThresholdInE = thePixelThreshold * theNoiseInElectrons; 


    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ){
       LogDebug ("PixelDigitizer") << " PixelDigitizer "  
				   << numColumns << " " << numRows << " " << moduleThickness;
    //MP DA SCOMMENTARE
       //        << thePixelThreshold << " " << thePixelThresholdInE << " " 
       // 	   << noiseInADCCounts << " " << theNoiseInElectrons << " ";
    }
    // produce SignalPoint's for all SimHit's in detector
    // Loop over hits
 
    vector<PSimHit>::const_iterator ssbegin; 
    for (ssbegin= _PixelHits.begin();ssbegin !=_PixelHits.end(); ++ssbegin) {
         if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
	LogDebug ("Pixel Digitizer") << (*ssbegin).particleType() << " " << (*ssbegin).pabs() << " " 
				     << (*ssbegin).energyLoss() << " " << (*ssbegin).tof() << " " 
				     << (*ssbegin).trackId() << " " << (*ssbegin).processType() << " " 
				     << (*ssbegin).detUnitId()  
				     << (*ssbegin).entryPoint() << " " << (*ssbegin).exitPoint() ; 
      }

      _collection_points.clear();  // Clear the container
      // fill _collection_points for this SimHit, indpendent of topology
      primary_ionization(*ssbegin); // fills _ionization_points

      drift(*ssbegin);  // transforms _ionization_points to _collection_points  

      // compute induced signal on readout elements and add to _signal
      induce_signal(*ssbegin); //*ihit needed only for SimHit<-->Digi link
				   }

    if(addNoise) add_noise();  // generate noise
    // Do only if needed 

    if((pixelInefficiency>0) && (_signal.size()>0)) 
      pixel_inefficiency(); // Kill some pixels

  }

  make_digis();
  return internal_coll;
}


//***********************************************************************/
// Generate primary ionization along the track segment. 
// Divide the track into small sub-segments  
void SiPixelDigitizerAlgorithm::primary_ionization(const PSimHit& hit) {

// // Straight line approximation for trajectory inside active media

  const float SegmentLength = 0.0010; //10microns in cm
  float energy;

  // Get the 3D segment direction vector 
  LocalVector direction = hit.exitPoint() - hit.entryPoint(); 

  float eLoss = hit.energyLoss();  // Eloss in GeV
  float length = direction.mag();  // Track length in Silicon

  NumberOfSegments = int ( length / SegmentLength); // Number of segments
  if(NumberOfSegments < 1)
    NumberOfSegments = 1;

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    LogDebug ("Pixel Digitizer") << " enter primary_ionzation " << NumberOfSegments 
				 << " shift = " 
				 << (hit.exitPoint().x()-hit.entryPoint().x()) << " " 
				 << (hit.exitPoint().y()-hit.entryPoint().y()) << " " 
				 << (hit.exitPoint().z()-hit.entryPoint().z()) << " "
				 << hit.particleType() <<" "<< hit.pabs() ; 
  }


  float* elossVector = new float[NumberOfSegments];  // Eloss vector

  if( fluctuateCharge ) {
    //MP DA RIMUOVERE ASSOLUTAMENTE
    //    int pid = hit.particleType();
    int pid=13;

    
    float momentum = hit.pabs();
    // Generate fluctuated charge points
    fluctuateEloss(pid, momentum, eLoss, length, NumberOfSegments, 
		   elossVector);
  }
  
  _ionization_points.resize( NumberOfSegments); // set size

//   // loop over segments
  for ( int i = 0; i != NumberOfSegments; i++) {
    // Divide the segment into equal length subsegments 
    Local3DPoint point = hit.entryPoint() + 
      float((i+0.5)/NumberOfSegments) * direction;

    if( fluctuateCharge ) 
      energy = elossVector[i]/GeVperElectron; // Convert charge to elec.
    else
      energy = hit.energyLoss()/GeVperElectron/float(NumberOfSegments);
    
    EnergyDepositUnit edu( energy, point); //define position,energy point
    _ionization_points[i] = edu; // save
    
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
      LogDebug ("Pixel Digitizer") << i << " " << _ionization_points[i].x() << " " 
				   << _ionization_points[i].y() << " " 
				   << _ionization_points[i].z() << " " 
				   << _ionization_points[i].energy();
     }
  }

  delete[] elossVector;

}

void SiPixelDigitizerAlgorithm::fluctuateEloss(int pid, float particleMomentum, 
				      float eloss, float length, 
				      int NumberOfSegs,float elossVector[]) {

  // Get dedx for this track
  float dedx;
  if( length > 0.) dedx = eloss/length;
  else dedx = eloss;



  // This is a big! apporximation. Needs to be improved.
  //const float zMaterial = 14.; // Fix to Silicon
  //particleMomentum = 2.; // Assume 2Gev/c

  double particleMass = 139.57; // Mass in MeV, Assume pion
  //MP DA RIMUOVERE
//   if( particleTable->getParticleData(pid) ) {  // Get mass from the PDTable
//     particleMass = 1000. * particleTable->getParticleData(pid)->mass(); //Conv. GeV to MeV
//   }

  //  pid = abs(pid);
  //if(pid==11) particleMass = 0.511;         // Mass in MeV
  //else if(pid==13) particleMass = 105.658;
  //else if(pid==211) particleMass = 139.570;
  //else if(pid==2212) particleMass = 938.271;

  // What is the track segment length.
  float segmentLength = length/NumberOfSegs;

  // Generate charge fluctuations.
  float de=0.;
  float sum=0.;
  double segmentEloss = (1000.*eloss)/NumberOfSegs; //eloss in MeV
  for (int i=0;i<NumberOfSegs;i++) {
    //       material,*,   momentum,energy,*, *,  mass
    //myglandz_(14.,segmentLength,2.,2.,dedx,de,0.14);
    // The G4 routine needs momentum in MeV, mass in Mev, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV 
    // Returns fluctuated eloss in MeV
    double deltaCutoff = tMax; // the cutoff is sometimes redefined inside, so fix it.
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

void SiPixelDigitizerAlgorithm::drift(const PSimHit& hit){

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
    LogDebug ("Pixel Digitizer") << " enter drift " ;
    }
  
  _collection_points.resize( _ionization_points.size()); // set size

  LocalVector driftDir=DriftDirection();


  if(driftDir.z() ==0.) {
    LogWarning("Magnetic field") << " pxlx: drift in z is zero ";
    return;
  }

  float TanLorenzAngleX = driftDir.x()/driftDir.z(); // tangen of Lorentz angle
  float TanLorenzAngleY = 0.; // force to 0, driftDir.y()/driftDir.z();
  // I also need cosines to estimate the path length
  float CosLorenzAngleX = 1./sqrt(1.+TanLorenzAngleX*TanLorenzAngleX); //cosine
  float CosLorenzAngleY = 1.;
 

   if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
      LogDebug ("Pixel Digitizer") << " Lorentz Tan " << TanLorenzAngleX << " " << TanLorenzAngleY <<" "
				   << CosLorenzAngleX << " " << CosLorenzAngleY << " "
				   << moduleThickness*TanLorenzAngleX << " " << driftDir;
    }

  float Sigma_x = 1.;  // Charge spread 
  float Sigma_y = 1.;
  float DriftDistance; // Distance between charge generation and collection 
  float DriftLength;   // Actual Drift Lentgh
  float Sigma;


  for (unsigned int i = 0; i != _ionization_points.size(); i++) {
        
    float SegX, SegY, SegZ; // position
    SegX = _ionization_points[i].x();
    SegY = _ionization_points[i].y();
    SegZ = _ionization_points[i].z();

    // Distance from the collection plane
    // Change drift direction to compensate for the tilt/cooridinate problem
    //    DriftDistance = (moduleThickness/2. - SegZ); // Drift to +z
    DriftDistance = (moduleThickness/2. + SegZ); // Drift to -z
    
    if( DriftDistance < 0.)
      DriftDistance = 0.;
    else if ( DriftDistance > moduleThickness )
      DriftDistance = moduleThickness;
    
    // Assume full depletion now, partial depletion will come later.
    float XDriftDueToMagField = DriftDistance * TanLorenzAngleX;
    float YDriftDueToMagField = DriftDistance * TanLorenzAngleY;
    
    // Shift cloud center
    float CloudCenterX = SegX + XDriftDueToMagField;
    float CloudCenterY = SegY + YDriftDueToMagField;

    // Calculate how long is the charge drift path
    DriftLength = sqrt( DriftDistance*DriftDistance + 
                        XDriftDueToMagField*XDriftDueToMagField +
                        YDriftDueToMagField*YDriftDueToMagField );

    // What is the charge diffusion after this path
    Sigma = sqrt(DriftLength/Dist300) * Sigma0;

    // Project the diffusion sigma on the collection plane
    Sigma_x = Sigma / CosLorenzAngleX ;
    Sigma_y = Sigma / CosLorenzAngleY ;

    SignalPoint sp( CloudCenterX, CloudCenterY,
     Sigma_x, Sigma_y, hit.tof(), _ionization_points[i].energy() );

    // Load the Charge distribution parameters
    _collection_points[i] = (sp);



  } // loop over ionization points, i.
 
}



void SiPixelDigitizerAlgorithm::induce_signal( const PSimHit& hit) {

  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)
  //DA MODIFICARE
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    LogDebug ("Pixel Digitizer") << " enter induce_signal, " 
				 << topol->pitch().first << " " << topol->pitch().second; //OK
  }

   // local map to store pixels hit by 1 Hit.      
   typedef map< int, float, less<int> > hit_map_type;
   hit_map_type hit_signal;

   // map to store pixel integrals in the x and in the y directions
   map<int, float, less<int> > x,y; 
   
   // Assign signals to readout channels and store sorted by channel number
   
   // Iterate over collection points on the collection plane
   for ( vector<SignalPoint>::const_iterator i=_collection_points.begin();
	 i != _collection_points.end(); i++) {
     
     float CloudCenterX = i->position().x(); // Charge position in x
     float CloudCenterY = i->position().y(); //                 in y
     float SigmaX = i->sigma_x();            // Charge spread in x
     float SigmaY = i->sigma_y();            //               in y
     float Charge = i->amplitude();          // Charge amplitude
     
 
     if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
       LogDebug ("Pixel Digitizer") << " cloud " << i->position().x() << " " << i->position().y() << " " 
				    << i->sigma_x() << " " << i->sigma_y() << " " << i->amplitude();
      
     }
     
     // Find the maximum cloud spread in 2D plane , assume 3*sigma
     float CloudRight = CloudCenterX + ClusterWidth*SigmaX;
     float CloudLeft  = CloudCenterX - ClusterWidth*SigmaX;
     float CloudUp    = CloudCenterY + ClusterWidth*SigmaY;
     float CloudDown  = CloudCenterY - ClusterWidth*SigmaY;
     
     // Define 2D cloud limit points
     LocalPoint PointRightUp  = LocalPoint(CloudRight,CloudUp);
     LocalPoint PointLeftDown = LocalPoint(CloudLeft,CloudDown);
     
     // Convert the 2D points to pixel indices

     MeasurementPoint mp = topol->measurementPosition(PointRightUp ); //OK
     
     int IPixRightUpX = int( floor( mp.x()));
     int IPixRightUpY = int( floor( mp.y()));

     if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) { 
       LogDebug ("Pixel Digitizer") << " right-up " << PointRightUp << " " 
				    << mp.x() << " " << mp.y() << " "
				    << IPixRightUpX << " " << IPixRightUpY ;
     }
 

     mp = topol->measurementPosition(PointLeftDown ); //OK
    
     int IPixLeftDownX = int( floor( mp.x()));
     int IPixLeftDownY = int( floor( mp.y()));
     
     if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) { 
       LogDebug ("Pixel Digitizer") << " left-down " << PointLeftDown << " " 
				    << mp.x() << " " << mp.y() << " "
				    << IPixLeftDownX << " " << IPixLeftDownY ;
     }

     // Check detector limits
     IPixRightUpX = numRows>IPixRightUpX ? IPixRightUpX : numRows-1 ;
     IPixRightUpY = numColumns>IPixRightUpY ? IPixRightUpY : numColumns-1 ;
     IPixLeftDownX = 0<IPixLeftDownX ? IPixLeftDownX : 0 ;
     IPixLeftDownY = 0<IPixLeftDownY ? IPixLeftDownY : 0 ;
     
     x.clear(); // clear temporary integration array
     y.clear();

     // First integrate cahrge strips in x
     int ix; // TT for compatibility
     for (ix=IPixLeftDownX; ix<=IPixRightUpX; ix++) {  // loop over x index
       float xUB, xLB, UpperBound, LowerBound;
      
       if (ix == 0) LowerBound = 0.;
       else {
	 mp = MeasurementPoint( float(ix), 0.0);
	 xLB = topol->localPosition(mp).x();
	 //	float oLowerBound = freq_( (xLB-CloudCenterX)/SigmaX);
	 
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xLB-CloudCenterX)/SigmaX, &result);

	 if (status != 0)  edm::LogWarning ("Integration")<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen";
	 LowerBound = 1-result.val;

       }
     
       if (ix == numRows-1) UpperBound = 1.;
       else {
	 mp = MeasurementPoint( float(ix+1), 0.0);
	 xUB = topol->localPosition(mp).x();
	 //	float oUpperBound = freq_( (xUB-CloudCenterX)/SigmaX);
	
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xUB-CloudCenterX)/SigmaX, &result);
	 if (status != 0)  edm::LogWarning ("Integration")<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen";

	UpperBound = 1. - result.val;

       }
       
       float   TotalIntegrationRange = UpperBound - LowerBound; // get strip
       x[ix] = TotalIntegrationRange; // save strip integral 
     }

    // Now integarte strips in y
    int iy; // TT for compatibility
    for (iy=IPixLeftDownY; iy<=IPixRightUpY; iy++) { //loope over y ind  
      float yUB, yLB, UpperBound, LowerBound;

      if (iy == 0) LowerBound = 0.;
      else {
        mp = MeasurementPoint( 0.0, float(iy) );
        yLB = topol->localPosition(mp).y();
	//	float oLowerBound = freq_( (yLB-CloudCenterY)/SigmaY);
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yLB-CloudCenterY)/SigmaY, &result);
	 if (status != 0)  edm::LogWarning ("Integration")<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen";

	LowerBound = 1. - result.val;



      }

      if (iy == numColumns-1) UpperBound = 1.;
      else {
        mp = MeasurementPoint( 0.0, float(iy+1) );
        yUB = topol->localPosition(mp).y();
	//        float oUpperBound = freq_( (yUB-CloudCenterY)/SigmaY);
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yUB-CloudCenterY)/SigmaY, &result);

	if (status != 0) edm::LogWarning ("Integration") <<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen";

	UpperBound = 1. - result.val;


      }

      float   TotalIntegrationRange = UpperBound - LowerBound;
      y[iy] = TotalIntegrationRange; // save strip integral
    }       

    // Get the 2D charge integrals by folding x and y strips
    int chan;
    for (ix=IPixLeftDownX; ix<=IPixRightUpX; ix++) {  // loop over x index
      for (iy=IPixLeftDownY; iy<=IPixRightUpY; iy++) { //loope over y ind  
	
        float ChargeFraction = Charge*x[ix]*y[iy];

        if( ChargeFraction > 0. ) {
	  chan = PixelDigi::pixelToChannel( ix, iy);  // Get index 

          // Load the amplitude						 
          hit_signal[chan] += ChargeFraction;
	} // endif


	if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) { 
	  mp = MeasurementPoint( float(ix), float(iy) );
	  LocalPoint lp = topol->localPosition(mp);
	  chan = topol->channel(lp);
	  LogDebug ("Pixel Digitizer") << " pixel " << ix << " " << iy << " - "<<" "
				       << chan << " " << ChargeFraction<<" "
				       << mp.x() << " " << mp.y() <<" "
				       << lp.x() << " " << lp.y() << " "  // givex edge position
				       << chan; // edge belongs to previous ?
	}
	
      } // endfor iy
    } //endfor ix
   
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      
      // Test conversions

      mp = topol->measurementPosition( i->position() ); //OK
      LocalPoint lp = topol->localPosition(mp);     //OK
      pair<float,float> p = topol->pixel( i->position() );  //OK
      chan = PixelDigi::pixelToChannel( int(p.first), int(p.second));
      pair<int,int> ip = PixelDigi::channelToPixel(chan);
      MeasurementPoint mp1 = MeasurementPoint( float(ip.first),
					       float(ip.second) );

      LogDebug ("Pixel Digitizer") << " Test "<< mp.x() << " " << mp.y() 
				   << " "<< lp.x() << " " << lp.y() << " "<<" "
				   <<p.first << " " << p.second << " "<< chan << " "
				   <<" " << ip.first << " " << ip.second << " "
				   << mp1.x() << " " << mp1.y() << " " //OK
				   << topol->localPosition(mp1).x() << " "  //OK
				   << topol->localPosition(mp1).y() << " "
				   << topol->channel( i->position() ); //OK
    }
    
  } // loop over charge distributions

  // Fill the global map with all hit pixels from this event
   
  for ( hit_map_type::const_iterator im = hit_signal.begin();
	im != hit_signal.end(); im++) {
    _signal[(*im).first] += Amplitude( (*im).second, &hit, (*im).second);
    
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      int chan =  (*im).first; 
      pair<int,int> ip = PixelDigi::channelToPixel(chan);
      LogDebug ("Pixel Digitizer") << " pixel " << ip.first << " " << ip.second << " "
				   << _signal[(*im).first];    
    }

  }

}

/***********************************************************************/
//
void SiPixelDigitizerAlgorithm::make_digis() {
  internal_coll.reserve(50); internal_coll.clear();

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    LogDebug ("Pixel Digitizer") << " make digis "<<" "
				 << " pixel threshold " << thePixelThresholdInE << " " 
				 << " List pixels passing threshold ";
  }

  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {
  
    float signalInElectrons = (*i).second ;   // signal in electrons
    // Do the miss calibration for calibration studies only.
    if(doMissCalibrate) signalInElectrons = missCalibrate(signalInElectrons);


    // Do only for pixels above threshold
    if ( signalInElectrons >= thePixelThresholdInE) {  
 
      int adc = int( signalInElectrons / theElectronPerADC ); // calibrate gain
      adc = min(adc, theAdcFullScale); // Check maximum value
       
     int chan =  (*i).first;  // channel number
      pair<int,int> ip = PixelDigi::channelToPixel(chan);
      if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
	LogDebug ("Pixel Digitizer") << (*i).first << " " << (*i).second << " " << signalInElectrons 
				     << " " << adc << ip.first << " " << ip.second ;
      }
      
      // Load digis
      internal_coll.push_back( PixelDigi( ip.first, ip.second, adc));
  





      //digilink     
     

      if((*i).second.hits().size()>0){
	simi.clear();
	unsigned int il=0;
	for( vector<const PSimHit*>::const_iterator ihit = (*i).second.hits().begin();
	     ihit != (*i).second.hits().end(); ihit++) {
	  simi[(**ihit).trackId()].push_back((*i).second.individualampl()[il]);
	  il++;
	}
	
	//sum the contribution of the same trackid 
	for( simlink_map::iterator simiiter=simi.begin();
	     simiiter!=simi.end();
	     simiiter++){
	  
	  float sum_samechannel=0;
	  for (unsigned int iii=0;iii<(*simiiter).second.size();iii++){
	    sum_samechannel+=(*simiiter).second[iii];
	  }
	  float fraction=sum_samechannel/(*i).second;
	  if (fraction>1.) fraction=1.;
	  link_coll.push_back(PixelDigiSimLink((*i).first,(*simiiter).first,fraction));
	}
	
      }
    }
   
  }
}

/***********************************************************************/
//
void SiPixelDigitizerAlgorithm::add_noise() {


  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    LogDebug ("Pixel Digitizer") << " enter add_noise " << theNoiseInElectrons;
  }

  // First add noise to hit pixels
  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {
    float noise  = RandGauss::shoot(0.,theNoiseInElectrons) ;
    (*i).second += Amplitude( noise,0,noise);
  
  }
  
  if(!addNoisyPixels)  // Option to skip noise in non-hit pixels
    return;

  // Add noise on non-hit pixels
  int numberOfPixels = (numRows * numColumns);

  map<int,float, less<int> > otherPixels;
  map<int,float, less<int> >::iterator mapI;
  
  theNoiser->generate(numberOfPixels, 
                      thePixelThreshold, //thr. in un. of nois
		      theNoiseInElectrons, // noise in elec. 
                      otherPixels );
  
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
    LogDebug ("Pixel Digitizer") <<  " Add noisy pixels " << numRows << " " << numColumns << " "
				 << theNoiseInElectrons << " " 
				 << thePixelThreshold << " " << numberOfPixels << " " 
				 << otherPixels.size() ;
  }

  for (mapI = otherPixels.begin(); mapI!= otherPixels.end(); mapI++) {
    int iy = ((*mapI).first) / numRows;
    int ix = ((*mapI).first) - (iy*numRows);

    // Keep for a while for testing.
    if( iy < 0 || iy > (numColumns-1) ) 
      LogWarning ("Pixel Geometry") << " error in iy " << iy ;
    if( ix < 0 || ix > (numRows-1) )
      LogWarning ("Pixel Geometry")  << " error in ix " << ix ;

    int chan = PixelDigi::pixelToChannel(ix, iy);

    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      LogDebug ("Pixel Digitizer")<<" Storing noise = " << (*mapI).first << " " << (*mapI).second 
				  << " " << ix << " " << iy << " " << chan ;
    }
  

    if(_signal[chan] == 0){
      //      float noise = float( (*mapI).second );
      int noise=int( (*mapI).second );
      _signal[chan] = Amplitude (noise, 0,noise);
    }
  }

}
/***********************************************************************/
//
void SiPixelDigitizerAlgorithm::pixel_inefficiency() {

  // Predefined efficiencies
  float pixelEfficiency  = 1.0;
  float columnEfficiency = 1.0;
  float chipEfficiency   = 1.0;

  // setup the chip indices conversion
  // At the moment I do not have a better way to find out the layer number? 
  unsigned int Subid=DetId(detID).subdetId();
  if    (Subid==  PixelSubdetector::PixelBarrel){// barrel layers
     int layerIndex=PXBDetId(detID).layer();
    pixelEfficiency  = thePixelEfficiency[layerIndex-1];
    columnEfficiency = thePixelColEfficiency[layerIndex-1];
    chipEfficiency   = thePixelChipEfficiency[layerIndex-1];
    
  } else {                // forward disks
  
    // For endcaps take same for each endcap
    pixelEfficiency  = thePixelEfficiency[3];
    columnEfficiency = thePixelColEfficiency[3];
    chipEfficiency   = thePixelChipEfficiency[3];

  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
    LogDebug ("Pixel Digitizer") << " enter pixel_inefficiency " << pixelEfficiency << " " 
				 << columnEfficiency << " " << chipEfficiency ;
  }

 
  PixelChipIndices indexConverter(52,80,
				  numColumns,numRows);

  int chipX,chipY,row,col;
  map<int, int, less<int> >chips, columns;
  map<int, int, less<int> >::iterator iter;
  
  // Find out the number of columns and chips hits
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for (signal_map_iterator i = _signal.begin();i != _signal.end();i++) {
    
    int chan = i->first;
    pair<int,int> ip = PixelDigi::channelToPixel(chan);
    int pixX = ip.first + 1;  // my indices start from 1
    int pixY = ip.second + 1;
    indexConverter.chipIndices(pixX,pixY,chipX,chipY,row,col);
   
    int chipIndex = indexConverter.chipIndex(chipX,chipY);
    pair<int,int> dColId = indexConverter.dColumn(row,col);
    //    int pixInChip  = dColId.first;
    int dColInChip = dColId.second;
    int dColInDet = indexConverter.dColumnIdInDet(dColInChip,chipIndex);
  
 
    
    chips[chipIndex]++;
    columns[dColInDet]++;
  }
  
 
  for ( iter = chips.begin(); iter != chips.end() ; iter++ ) {
 
    float rand  = RandFlat::shoot();
    if( rand > chipEfficiency ) chips[iter->first]=0;
 
  }
 
  for ( iter = columns.begin(); iter != columns.end() ; iter++ ) {
 
    float rand  = RandFlat::shoot();
    if( rand > columnEfficiency ) columns[iter->first]=0;
 
  }
  
 
  // Now loop again over pixel to kill some
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {
    

    pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
    int pixX = ip.first + 1;  // my indices start from 1
    int pixY = ip.second + 1;
    
    indexConverter.chipIndices(pixX,pixY,chipX,chipY,row,col); //get chip index
    int chipIndex = indexConverter.chipIndex(chipX,chipY);
    
    pair<int,int> dColId = indexConverter.dColumn(row,col);  // get dcol index
    int dColInDet = indexConverter.dColumnIdInDet(dColId.second,chipIndex);




    float rand  = RandFlat::shoot();

   if( chips[chipIndex]==0 ||  columns[dColInDet]==0 
	|| rand>pixelEfficiency ) {
      // make pixel amplitude =0, pixel will be lost at clusterization    
      i->second.set(0.); // reset amplitude, 
   
    }


  }
  
}
//***********************************************************************
// Fluctuate the gain and offset for the amplitude calibration
// Use gaussian smearing.
float SiPixelDigitizerAlgorithm::missCalibrate(const float amp) const {
  float gain  = RandGauss::shoot(1.,theGainSmearing);
  float offset  = RandGauss::shoot(0.,theOffsetSmearing);
  float newAmp = amp * gain + offset;
  return newAmp;
}  
LocalVector SiPixelDigitizerAlgorithm::DriftDirection(){
  //good Drift direction estimation only for pixel barrel
  Frame detFrame(_detp->surface().position(),_detp->surface().rotation());
  LocalVector Bfield=detFrame.toLocal(_bfield);
  //  if    (DetId(detID).subdetId()==  PixelSubdetector::PixelBarrel){
    float dir_x = tanLorentzAnglePerTesla * Bfield.y();
    float dir_y = -tanLorentzAnglePerTesla * Bfield.x();
    float dir_z = 1.; // E field always in z direction
    LocalVector theDriftDirection = LocalVector(dir_x,dir_y,dir_z);
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      LogDebug ("Pixel Digitizer") << " The drift direction in local coordinate is "   
				   << theDriftDirection ;
    }
    return theDriftDirection;
 
}

