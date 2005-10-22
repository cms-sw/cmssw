//class SiStripDigitizerAlgorithm SimTracker/SiStripDigitizer/src/SiStripDigitizerAlgorithm.cc

// Ported in CMSSW by  Andrea Giammanco, following the structure by Michele Pioppi-INFN perugia
//         Created:  Mon Sep 26 11:08:32 CEST 2005



#include <vector>
#include <iostream>
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"
#include <gsl/gsl_sf_erf.h>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
//#include "SimTracker/SiStripDigitizer/interface/StripChipIndices.h"

using namespace std;

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf):conf_(conf){
  digis=0;

  // Common strip parameters
  // This are parameters which are not likely to be changed
  NumberOfSegments = 20; // Default number of track segment divisions
  ClusterWidth = 3.;     // Charge integration spread on the collection plane
  GeVperElectron = 3.7E-09;  // 1 electrons=3.7eV, 1keV=270.3e
  Sigma0 = 0.0007;           // Charge diffusion constant 
  Dist300 = 0.0300;          //   normalized to 300micron Silicon

  //get external parameters
  // ADC calibration 1adc count = 135e.
  // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc]=2[adc/kev]
  // Be carefull, this parameter is also used in SiStripDet.cc to 
  // calculate the noise in adc counts from noise in electrons.
  // Both defaults should be the same.
  theElectronPerADC=conf_.getParameter<double>("ElectronPerAdc");

  // ADC saturation value, 255=8bit adc.
  theAdcFullScale=conf_.getParameter<int>("AdcFullScale");

  // Strip threshold in units of noise.
  theStripThreshold=conf_.getParameter<double>("ThresholdInNoiseUnits");

  //theTofCut 12.5, cut in particle TOD +/- 12.5ns
  theTofCut=conf_.getParameter<double>("TofCut");

  // Add noise   
  addNoise=conf_.getParameter<bool>("AddNoise");

  // Add noisy strips 
  addNoisyStrips=conf_.getParameter<bool>("AddNoisyStrips");

  // Fluctuate charge in track subsegments
  fluctuateCharge=conf_.getParameter<bool>("FluctuateCharge");

  // delta cutoff in MeV, has to be same as in OSCAR=0.030/cmsim=1.0 MeV
  //tMax = 0.030; // In MeV.  
  tMax =conf_.getParameter<double>("DeltaProductionCut");  
 

  // Get the constants for the miss-calibration studies
  doMissCalibrate=conf_.getParameter<bool>("MissCalibrate"); // Enable miss-calibration
  theGainSmearing=conf_.getParameter<double>("GainSmearing"); // sigma of the gain smearing
  theOffsetSmearing=conf_.getParameter<double>("OffsetSmearing"); //sigma of the offset smearing

  //////strip geometry
  theStripsInChip=conf_.getParameter<int>("StripAPV");
  
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout<<"SiStripDigitizerAlgorithm constructed"<<endl;
    cout<<"Configuraion parameters:"<<endl;  
    cout << "Threshold/Gain = "  
	 << theStripThreshold << " " <<  theElectronPerADC 
	 << " " << theAdcFullScale << endl; 
    cout << " The delta cut-off is set to " << tMax << endl;
    if(doMissCalibrate) cout << " miss-calibrate the strip amplitude " 
			     << theGainSmearing << " " << theOffsetSmearing 
			     << endl;
  }
  //MP DA RISOLVERE
  // particleTable =  &HepPDT::theTable();

}
SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm(){
 if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
   cout<<"SiStripDigitizerAlgorithm deleted"<<endl;
 }
}

//void  SiStripDigitizerAlgorithm::run(const std::vector<PSimHit*> &input,StripDigiCollection &output)
void  SiStripDigitizerAlgorithm::run(const std::vector<PSimHit*> &input,StripDigiCollection &output,StripGeomDetUnit *stripdet)
{


  _detp = stripdet; //cache the StripGeomDetUnit

  // Strip Efficiency moved from the constructor to the method run because
  // the information of the det are nota available in the constructor
  // Effciency parameters. 0 - no inefficiency, 1-low lumi, 10-high lumi
  // enum StripGeomDetType::SubDetector stripPart;

  stripPart=stripdet->type().subDetector();


  theStripLuminosity=conf_.getParameter<int>("AddStripInefficiency");
  if (theStripLuminosity>0) {
    stripInefficiency=true;
    // Default efficiencies 
    for (int i=0; i<3;i++) { // Andrea: sostituire 3 con il numero di layers
      // Assume 1% inefficiency for single strips, 
      // this is given by faulty bump-bonding and seus.  
      theStripEfficiency[i]     = 1.-0.01;  // strips = 99%
      // A flat 0.25% inefficiency due to lost data packets from TBM
      theStripChipEfficiency[i] = 1.-0.0025; // chips = 99.75%
    }
    
    
    
    // Special cases 

    //MP DA VERIFICARE il valore di strippart
//     if( stripPart == barrel ) {  // For the barrel
    if( stripPart == 0 ) {  // For the barrel
      if(theStripLuminosity==10) { // For high luminosity
	theStripEfficiency[0]    = 1.-0.015; // 1.5% for r=4
      }        
    }



    // Set efficencies to a preset values (Testing only),-1=not used(def)
    StripEff=conf_.getParameter<double>("StripEfficiency");
    StripChipEff=conf_.getParameter<double>("StripChipEfficiency");

    if(StripEff>0.) {     // Set all layers to the preset value
      for (int i=0; i<3;i++) {
	theStripEfficiency[i] = StripEfficiency;
      }
    }
    if(StripChipEff>0.) {
      for (int i=0; i<3;i++) {
	theStripChipEfficiency[i] = StripChipEfficiency;
      }
    }
  }
  
  //MP QUESTA PARTE VA COMPLETAMENTE MODIFICATA NON APPENA SARANNO DISPONIBILI 
  // I SIMHIT

  unsigned int detID = 0;
  unsigned int newDetID = 0;

  int detunits = 0;
  bool first = true;
  //  vector<StripDigi> collector;
  ss.clear();
  //raggruppati gli hit in base alla detunit
  
  
  vector<PSimHit*>::const_iterator simHitIter = input.begin();
  vector<PSimHit*>::const_iterator simHitIterEnd = input.end();
  //  vector<PseudoHit*>::const_iterator simHitIter = input.begin();
  //  vector<PseudoHit*>::const_iterator simHitIterEnd = input.end();
  //start the loop over the simhits
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    cout << "# digis: " << digis << endl;
    ++digis;
    //pointer to the simhit
    const PSimHit *simHitPointer = *simHitIter;
  
 
    if ( first ) {
      detID = simHitPointer->detUnitId();
      first = false;
    }
    newDetID =  simHitPointer->detUnitId();
    if(detID==newDetID){
      ss.push_back(simHitPointer);     
    }

    
    //     /////////////////      
    
  }
  

  cout << "TEMP1: digis =" << digis << ", detunits+1 = " << detunits+1 << endl;
  //Digitization of the SimHits of a given stripdet
  vector<StripDigi> collector =digitize(stripdet);
  cout << "TEMP2: digis =" << digis << ", detunits+1 = " << detunits+1 << endl;
  

  //Fill the stripidigicollection
  StripDigiCollection::Range outputRange;
  outputRange.first = collector.begin();
  outputRange.second = collector.end();
  output.put(outputRange,detID);
  collector.clear();

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << "[SiStripDigitizerAlgorithm] converted " << digis << " StripDigis in " << detunits+1 << " DetUnits." << endl; 
  }
  
}
/**********************************************************************/

vector<StripDigi> SiStripDigitizerAlgorithm::digitize(StripGeomDetUnit *det){
  

  //  digis = 0; // commented out by Andrea
  if( ss.size() > 0 || addNoisyStrips) {
  
    topol=&det->specificTopology(); // cache topology
    numStrips = topol->nstrips();  // det module number of strips


    moduleThickness = det->specificSurface().bounds().thickness(); // full detector thicness

    //MP DA SISTEMARE
    //     float noiseInADCCounts = _detp->readout().noiseInAdcCounts();
    float noiseInADCCounts=3.7;  
    // For the noise generation I need noise in electrons
    theNoiseInElectrons = noiseInADCCounts * theElectronPerADC;    
    // Find the threshold in electrons
    theStripThresholdInE = theStripThreshold * theNoiseInElectrons; 


    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ){
       cout << " StripDigitizer: stripPart " << stripPart << ", numStrips " 
	    << numStrips << ", moduleThickness " << moduleThickness<<endl;
    //MP DA SCOMMENTARE
//       cout << theStripThreshold << " " << theStripThresholdInE << " " 
// 	   << noiseInADCCounts << " " << theNoiseInElectrons << endl;
    }
    // produce SignalPoint's for all SimHit's in detector
    // Loop over hits
    vector<const PSimHit*>::const_iterator ssbegin = ss.begin();
    vector<const PSimHit*>::const_iterator ssend = ss.end();
    for (;ssbegin != ssend; ++ssbegin) {
      const PSimHit *pointerHit = *ssbegin;
 

      if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
	cout << "particle type: " << pointerHit->particleType() << ", pabs: " << pointerHit->pabs() << " " 
	     << pointerHit->energyLoss() << ", eloss: " << pointerHit->tof() << ", detUnit ID: " 
	  //MP packedTrackId???    
	  //	  << pointerHit->packedTrackId() << " " << pointerHit->processType() << " " 
	     << pointerHit->detUnitId() << endl; 
	cout << "entry point: " << pointerHit->entryPoint() << ", exit point: " << pointerHit->exitPoint() << endl; 
      }


      _collection_points.clear();  // Clear the container
      // fill _collection_points for this SimHit, indpendent of topology
      primary_ionization(*pointerHit); // fills _ionization_points
      drift(*pointerHit);  // transforms _ionization_points to _collection_points  

      // compute induced signal on readout elements and add to _signal
      induce_signal(*pointerHit); //*ihit needed only for SimHit<-->Digi link


				      //      int adc=10;
    }
  
    if(addNoise) add_noise();  // generate noise

    // Do only if needed 
    if(stripInefficiency && _signal.size()>0 ) 
      strip_inefficiency(); // Kill some strips
  }
  make_digis();
  return internal_coll;
}


/***********************************************************************/
// Generate primary ionization along the track segment. 
// Divide the track into small sub-segments  
void SiStripDigitizerAlgorithm::primary_ionization(const PSimHit& hit) {

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
    cout << " enter primary_ionization " << NumberOfSegments << " " ;
    cout << " shift = " 
	 << (hit.exitPoint().x()-hit.entryPoint().x()) << " " 
	 << (hit.exitPoint().y()-hit.entryPoint().y()) << " " 
	 << (hit.exitPoint().z()-hit.entryPoint().z()) << " "
	 << hit.particleType() <<" "<< hit.pabs() 
	 << endl; 
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
    
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 2 ) {  
      cout << i << " " << _ionization_points[i].x() << " " 
	   << _ionization_points[i].y() << " " 
	   << _ionization_points[i].z() << " " 
	   << _ionization_points[i].energy() <<endl;
     }
  }

  delete[] elossVector;

}

void SiStripDigitizerAlgorithm::fluctuateEloss(int pid, float particleMomentum, 
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
    //cout << " fluctuate charge, ratio = " << ratio <<" "<< eloss <<" "
    // << sum <<" "<< length<<" " <<dedx<<" "<<NumberOfSegs<<" " 
    // << tMax << " " << particleMass << endl;
    for (int ii=0;ii<NumberOfSegs;ii++) elossVector[ii]= ratio*elossVector[ii];
  } else {  // If fluctuations gives 0 eloss
    float averageEloss = eloss/NumberOfSegs;
    for (int ii=0;ii<NumberOfSegs;ii++) elossVector[ii]= averageEloss; 
  }
  return;
}

void SiStripDigitizerAlgorithm::drift(const PSimHit& hit){

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
    cout << " enter drift " << endl;
  }
  
  _collection_points.resize( _ionization_points.size()); // set size

  LocalPoint center(0.,0.);  // detector center 

  //MP dove e' driftdirection?
  // LocalVector driftDir = _detp->driftDirection(center); // drift in center
  LocalVector driftDir(1.,1.,1.);


  if(driftDir.z() ==0.) {
    cout << " strip: drift in z is zero " << endl;
    return;
  }

  float TanLorenzAngleX = driftDir.x()/driftDir.z(); // tangen of Lorentz angle
  float TanLorenzAngleY = 0.; // force to 0, driftDir.y()/driftDir.z();
  // I also need cosines to estimate the path length
  float CosLorenzAngleX = 1./sqrt(1.+TanLorenzAngleX*TanLorenzAngleX); //cosine
  float CosLorenzAngleY = 1.;
 

   if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
      cout << " Lorentz Tan " << TanLorenzAngleX << " " << TanLorenzAngleY <<" "
	   << CosLorenzAngleX << " " << CosLorenzAngleY << " "
	   << moduleThickness*TanLorenzAngleX << " " << driftDir << endl;
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



void SiStripDigitizerAlgorithm::induce_signal( const PSimHit& hit) {

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << " enter induce_signal, pitch = " 
	 << topol->pitch() << endl; //Modified by AG for strips
  }

   // local map to store strips hit by 1 Hit.      
  typedef map< int, float, less<int> > hit_map_type;
  hit_map_type hit_signal;


  // map to store strip integrals in the x and in the y directions
  map<int, float, less<int> > x,y; 

  // Assign signals to readout channels and store sorted by channel number
   
  // Iterate over collection points on the collection plane
  for ( vector<SignalPoint>::const_iterator i=_collection_points.begin();
	i != _collection_points.end(); i++) {
     
    float CloudCenterX = i->position().x(); // Charge position in x
    //    float CloudCenterY = i->position().y(); //                 in y
    float SigmaX = i->sigma_x();            // Charge spread in x
    //    float SigmaY = i->sigma_y();            //               in y
    float Charge = i->amplitude();          // Charge amplitude
     
 
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 2 ) {  
      cout << " cloud " << i->position().x() << " " << i->position().y() << " " 
	   << i->sigma_x() << " " << i->sigma_y() << " " << i->amplitude() <<
	endl;
      //    cout << CloudCenterX << " " << CloudCenterY << " " <<
      //      SigmaX << " " << SigmaY << " " << Charge << " ";
     }
     
    // Find the maximum cloud spread in 1D (AG) , assume 3*sigma
    float CloudRight = CloudCenterX + ClusterWidth*SigmaX;
    float CloudLeft  = CloudCenterX - ClusterWidth*SigmaX;
    // float CloudUp    = CloudCenterY + ClusterWidth*SigmaY;
    // float CloudDown  = CloudCenterY - ClusterWidth*SigmaY;
  
  
     
     // Define 2D cloud limit points
     //     LocalPoint PointRightUp  = LocalPoint(CloudRight,CloudUp);
     //     LocalPoint PointLeftDown = LocalPoint(CloudLeft,CloudDown);
    LocalPoint PointRight  = LocalPoint(CloudRight,0.);
    LocalPoint PointLeft = LocalPoint(CloudLeft,0.);
     
     // Convert the 2D points to strip indices

    //     MeasurementPoint mp = topol->measurementPosition(PointRightUp ); //OK
    MeasurementPoint mp = topol->measurementPosition(PointRight); //(AG)
     
    //    int IStripRightUpX = int( floor( mp.x()));
    //    int IStripRightUpY = int( floor( mp.y()));
    int IStripRight = int( floor( mp.x()));

    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 2 ) { 
      cout << " right " << PointRight << " " ;
      cout << mp.x() << " ";
      cout << IStripRight << endl;
    }
 

    //    mp = topol->measurementPosition(PointLeftDown ); //OK
    mp = topol->measurementPosition(PointLeft ); //(AG)
    
    //     int IStripLeftDownX = int( floor( mp.x()));
    //     int IStripLeftDownY = int( floor( mp.y()));
    int IStripLeft = int( floor( mp.x()));
     
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 2 ) { 
      cout << " left " << PointLeft << " " ;
      cout << mp.x() << " ";
      cout << IStripLeft << endl;
    }

     // Check detector limits (1D case, AG)
    IStripRight = numStrips>IStripRight ? IStripRight : numStrips-1;
    IStripLeft = 0<IStripLeft ? IStripLeft : 0;
    /*
     IStripRightUpX = numRows>IStripRightUpX ? IStripRightUpX : numRows-1 ;
     IStripRightUpY = numColumns>IStripRightUpY ? IStripRightUpY : numColumns-1 ;
     IStripLeftDownX = 0<IStripLeftDownX ? IStripLeftDownX : 0 ;
     IStripLeftDownY = 0<IStripLeftDownY ? IStripLeftDownY : 0 ;
     
     x.clear(); // clear temporary integration array
     y.clear();
    */

     // First integrate charge strips in x
     int ix; // TT for compatibility
     //     for (ix=IStripLeftDownX; ix<=IStripRightUpX; ix++) {  // loop over x index
     for (ix=IStripLeft; ix<=IStripRight; ix++) {  // 1D (AG)
       float xUB, xLB, UpperBound, LowerBound;
      
       if (ix == 0) LowerBound = 0.;
       else {
	 mp = MeasurementPoint( float(ix), 0.0);
	 xLB = topol->localPosition(mp).x();
	 //	float oLowerBound = freq_( (xLB-CloudCenterX)/SigmaX);
	 
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xLB-CloudCenterX)/SigmaX, &result);
	 //MP da verificare
	 //	 if (status != 0) throw DetLogicError("GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen");
	 if (status != 0)  cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<endl;
	 LowerBound = 1-result.val;
	 //	cout <<" LOWERB STRIP "<<oLowerBound<<" " <<LowerBound<<" " <<oLowerBound-LowerBound<<endl;
       }
     
       if (ix == numStrips-1) UpperBound = 1.;
       else {
	 mp = MeasurementPoint( float(ix+1), 0.0);
	 xUB = topol->localPosition(mp).x();
	 //	float oUpperBound = freq_( (xUB-CloudCenterX)/SigmaX);
	
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xUB-CloudCenterX)/SigmaX, &result);
	 //MP da verificare
	 if (status != 0)  cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<endl;
	 //	 if (status != 0) throw DetLogicError("GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen");
	 UpperBound = 1. - result.val;
	//	cout <<" LOWERB STRIP "<<oUpperBound<<" " <<UpperBound<<" " <<oUpperBound-UpperBound<<endl;

       }
       
       float   TotalIntegrationRange = UpperBound - LowerBound; // get strip
       x[ix] = TotalIntegrationRange; // save strip integral 
       if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 2 ) { 
	 cout << " TotalIntegrationRange =  " << UpperBound << " - " << LowerBound << " = " << TotalIntegrationRange << endl ;
	 cout << " Cloud:  " << CloudLeft << " " << CloudRight << endl ;
       }

     }

  
     // Get the 1D charge integrals (AG)
     int chan;
     for (ix=IStripLeft; ix<=IStripRight; ix++) {  // 1D (AG)
       float ChargeFraction = Charge*x[ix];

       if( ChargeFraction > 0. ) {
	 //	  chan = StripDigi::stripToChannel( ix, iy);  // Get index 

	 // Load the amplitude						 
	 //	 hit_signal[chan] += ChargeFraction;
	 hit_signal[ix] += ChargeFraction;
       } // endif

     } //endfor ix

     /*
	if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) { 
	  cout << " strip " << ix << " " << iy << " - ";
	  cout << chan << " " << ChargeFraction << endl; //OK
	  mp = MeasurementPoint( float(ix), float(iy) );
	  cout << mp.x() << " " << mp.y() << " "; //OK
	  LocalPoint lp = topol->localPosition(mp);
	  cout << lp.x() << " " << lp.y() << " ";  // givex edge position
	  chan = topol->channel(lp); // something wrong 1->0, 
	  cout << chan << endl; // edge belongs to previous ?
	}
     */
  
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 2 ) {
      
      // Test conversions
      cout << " Test " << endl;
      mp = topol->measurementPosition( i->position() ); //OK
      cout << mp.x() << " " << mp.y() << " ";
      LocalPoint lp = topol->localPosition(mp);     //OK
      cout << lp.x() << " " << lp.y() << " ";
      float p = topol->strip( i->position() );  // AG for the 1D case
      cout << p << " " << endl; // AG for the 1D case
    }


  } // loop over charge distributions

  // Fill the global map with all hit strips from this event
  for ( hit_map_type::const_iterator im = hit_signal.begin();
	im != hit_signal.end(); im++) {
    _signal[(*im).first] += Amplitude( (*im).second, &hit);

    /*
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      int chan =  (*im).first; 
      pair<int,int> ip = StripDigi::channelToStrip(chan);
      cout << " strip " << ip.first << " " << ip.second << " ";
      //    cout << (*im).first << " " << (*im).second << " ";    
      cout << _signal[(*im).first] << endl;    
    }
    */

  }

}

/***********************************************************************/
//
void SiStripDigitizerAlgorithm::make_digis() {
  internal_coll.reserve(50); internal_coll.clear();

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << " make digis ";
    cout << " strip threshold " << theStripThresholdInE << endl; 
    cout << " List strips passing threshold " << endl;
  }

  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {

    float signalInElectrons = (*i).second ;   // signal in electrons

    // Do the miss calibration for calibration studies only.
    if(doMissCalibrate) signalInElectrons = missCalibrate(signalInElectrons);


    // Do only for strips above threshold
    if ( signalInElectrons >= theStripThresholdInE) {  

      int adc = int( signalInElectrons / theElectronPerADC ); // calibrate gain
      adc = min(adc, theAdcFullScale); // Check maximum value
       
      
      int chan =  (*i).first;  // channel number
      int ip = chan;
      //      pair<int,int> ip = StripDigi::channelToStrip(chan);
      //      if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      //	cout << (*i).first << " " << (*i).second << " " << signalInElectrons 
      //      	     << " " << adc << ip.first << " " << ip.second << endl;
      //      }
      
      // Load digis
      internal_coll.push_back( StripDigi( ip, adc));

   
      // Find the right Hit id
      for( vector<const PSimHit*>::const_iterator ihit = (*i).second.hits().begin();
	   ihit != (*i).second.hits().end(); ihit++) {

	//MP DA COMPLETARE	
	// 	if(*ihit) { // do only for valid hits
// 	  //                                why is the fraction  always 1.?
// 	  _detp->simDet()->addLink( (*i).first, (**ihit).packedTrackId(),1.);
// 	}
      }
    }
  }

   
}

/***********************************************************************/
//
void SiStripDigitizerAlgorithm::add_noise() {
  
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << " enter add_noise " << theNoiseInElectrons << endl;
  }

  // First add noise to hit strips
  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {
    float noise  = RandGauss::shoot(0.,theNoiseInElectrons) ;
    (*i).second += Amplitude( noise,0);
    //     cout << (*i).first << " " << (*i).second << " ";
    //     cout << noise << " " << (*i).second << endl;
  }
  
  if(!addNoisyStrips)  // Option to skip noise in non-hit strips
    return;

  // Add noise on non-hit strips
  //  int numberOfStrips = (numRows * numColumns);
  int numberOfStrips = numStrips;

  map<int,float, less<int> > otherStrips;
  map<int,float, less<int> >::iterator mapI;

  
  theNoiser->generate(numberOfStrips, 
                      theStripThreshold, //thr. in un. of nois
		      theNoiseInElectrons, // noise in elec. 
                      otherStrips );

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
    cout <<  " Add noisy strips " << numStrips << " "
	 << theNoiseInElectrons << " " 
	 << theStripThreshold << " " << numberOfStrips << " " 
	 << otherStrips.size() << endl;
  }

  for (mapI = otherStrips.begin(); mapI!= otherStrips.end(); mapI++) {
    cout << "sono entrato qui" << endl;

  ////////// Andrea //////////////
  /*
    
    int iy = ((*mapI).first) / numStrips;
    int ix = ((*mapI).first) - (iy*numStrips);

    // Keep for a while for testing.
    if( iy < 0 || iy > (numColumns-1) ) 
      cout << " error in iy " << iy << endl;
    if( ix < 0 || ix > (numRows-1) )
      cout << " error in ix " << ix << endl;

    int chan = StripDigi::stripToChannel(ix, iy);

    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      cout<<" Storing noise = " << (*mapI).first << " " << (*mapI).second 
	  << " " << ix << " " << iy << " " << chan <<endl;
    }
  */  
    
    int chan = (*mapI).first;
    if(_signal[chan] == 0){
      float noise = float( (*mapI).second );
      _signal[chan] = Amplitude (noise, 0);
    }
  }

}
/***********************************************************************/
//
void SiStripDigitizerAlgorithm::strip_inefficiency() {
  //// questa parte e' per i pixel!!!!
  /// aggiornarla per le strip (ANDREA)


  // Predefined efficiencies
  float stripEfficiency  = 1.0;
  float chipEfficiency   = 1.0;

  // setup the chip indices conversion
  // At the moment I do not have a better way to find out the layer number? 
  //MP da verificare
  //  if ( stripPart == barrel ) {  // barrel layers
  if ( stripPart == 0 ) {  // barrel layers
    //MP da capire
    //    double radius = _detp->position().perp();
    double radius = _detp->surface().position().perp();
    
    int layerIndex = 0;
    if( radius < 5.5 ) {
      layerIndex=1;
    } else if ( radius < 9. ) {
      layerIndex=2;
    } else  {
      layerIndex=3;
    }
    
    stripEfficiency  = theStripEfficiency[layerIndex-1];
    chipEfficiency   = theStripChipEfficiency[layerIndex-1];
    
  } else {                // forward disks
    
    //double zabs = fabs( _detp->position().z() );
    //int layerIndex = 0;
    //if( zabs < 40. ) {
    //  layerIndex=1;
    //} else if ( zabs < 50. ) {
    //  layerIndex=2;
    //} else  {
    //  layerIndex=3;
    //}

    // For endcaps take same for each endcap
    stripEfficiency  = theStripEfficiency[0];
    chipEfficiency   = theStripChipEfficiency[0];

  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
    cout << " enter strip_inefficiency " << stripEfficiency << " " 
	 << chipEfficiency << endl;
  }
  
  // Initilize the index converter
  //  StripChipIndices indexConverter(theStripsInChip,
  //				  numStrips);

  /////////// Andrea /////////////
  /*
  int chipX,chipY,row,col;
  map<int, int, less<int> >chips, columns;
  map<int, int, less<int> >::iterator iter;
  
  // Find out the number of columns and chips hits
  // Loop over hit strips, amplitude in electrons, channel = coded row,col
  for (signal_map_iterator i = _signal.begin();i != _signal.end();i++) {
    
    int chan = i->first;
    pair<int,int> ip = StripDigi::channelToStrip(chan);
    int stripX = ip.first + 1;  // my indices start from 1
    int stripY = ip.second + 1;
    
    indexConverter.chipIndices(stripX,stripY,chipX,chipY,row,col);
    int chipIndex = indexConverter.chipIndex(chipX,chipY);
    pair<int,int> dColId = indexConverter.dColumn(row,col);
    //    int stripInChip  = dColId.first;
    int dColInChip = dColId.second;
    int dColInDet = indexConverter.dColumnIdInDet(dColInChip,chipIndex);
    
    //cout <<chan << " " << stripX << " " << stripY << " " << (*i).second  << endl;
    //cout << chipX << " " << chipY << " " << row << " " << col << " " 
    // << chipIndex << " " << stripInChip << " " << dColInChip << " " 
    // << dColInDet << endl;
    
    chips[chipIndex]++;
    columns[dColInDet]++;
  }
  
  //cout << " chips hit " << chips.size() << endl;
  for ( iter = chips.begin(); iter != chips.end() ; iter++ ) {
    //cout << iter->first << " " << iter->second << endl;
    float rand  = RandFlat::shoot();
    if( rand > chipEfficiency ) chips[iter->first]=0;
    //cout << rand << " "  << iter->first << " " << iter->second << endl;
    //if( iter->second == 0 ) cout << " chip erased " << endl;
  }
  
  //cout << " columns hit " << columns.size() << endl;
  for ( iter = columns.begin(); iter != columns.end() ; iter++ ) {
    //cout << iter->first << " " << iter->second << endl;
    float rand  = RandFlat::shoot();
    if( rand > columnEfficiency ) columns[iter->first]=0;
    //cout << rand << " "  << iter->first << " " << iter->second << endl;
    //if( iter->second == 0 ) cout << " column erased " << endl;
  }
  
  //cout << " strip " << _signal.size() << endl;
  // Now loop again over strip to kill some
  // Loop over hit strips, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {
    
    //cout << " strip " << i->first << " " << float(i->second) << endl;    
    //    int chan = i->first;
    pair<int,int> ip = StripDigi::channelToStrip(i->first);//get strip pos
    int stripX = ip.first + 1;  // my indices start from 1
    int stripY = ip.second + 1;
    
    indexConverter.chipIndices(stripX,stripY,chipX,chipY,row,col); //get chip index
    int chipIndex = indexConverter.chipIndex(chipX,chipY);
    
    pair<int,int> dColId = indexConverter.dColumn(row,col);  // get dcol index
    int dColInDet = indexConverter.dColumnIdInDet(dColId.second,chipIndex);

    //cout <<chan << " " << stripX << " " << stripY << " " << (*i).second  << endl;
    //cout << chipX << " " << chipY << " " << row << " " << col << " " 
    // << chipIndex << " " << dColInDet << endl;
    //cout <<  chips[chipIndex] << " " <<  columns[dColInDet] << endl; 

    float rand  = RandFlat::shoot();
    if( chips[chipIndex]==0 ||  columns[dColInDet]==0 
	|| rand>stripEfficiency ) {
      // make strip amplitude =0, strip will be lost at clusterization    
      i->second.set(0.); // reset amplitude, 
      //cout << " strip will be killed " << float(i->second) << " " 
      //   << rand << " " << chips[chipIndex] << " " 
      //   << columns[dColInDet] << endl;
   }
  */
  
}

//***********************************************************************
// Fluctuate the gain and offset for the amplitude calibration
// Use gaussian smearing.
float SiStripDigitizerAlgorithm::missCalibrate(const float amp) const {
  float gain  = RandGauss::shoot(1.,theGainSmearing);
  float offset  = RandGauss::shoot(0.,theOffsetSmearing);
  float newAmp = amp * gain + offset;
  return newAmp;
}  
