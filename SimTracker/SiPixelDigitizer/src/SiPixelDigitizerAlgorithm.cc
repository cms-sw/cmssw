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

using namespace std;

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
  theElectronPerADC=conf_.getParameter<double>("ElectronPerAdc");

  // ADC saturation value, 255=8bit adc.
  theAdcFullScale=conf_.getParameter<int>("AdcFullScale");

  // Pixel threshold in units of noise.
  thePixelThreshold=conf_.getParameter<double>("ThresholdInNoiseUnits");

  //theTofCut 12.5, cut in particle TOD +/- 12.5ns
  theTofCut=conf_.getParameter<double>("TofCut");

  // Add noise   
  addNoise=conf_.getParameter<bool>("AddNoise");

  // Add noisy pixels 
  addNoisyPixels=conf_.getParameter<bool>("AddNoisyPixels");

  // Fluctuate charge in track subsegments
  fluctuateCharge=conf_.getParameter<bool>("FluctuateCharge");

  // delta cutoff in MeV, has to be same as in OSCAR=0.030/cmsim=1.0 MeV
  //tMax = 0.030; // In MeV.  
  tMax =conf_.getParameter<double>("DeltaProductionCut");  
 

  // Get the constants for the miss-calibration studies
  doMissCalibrate=conf_.getParameter<bool>("MissCalibrate"); // Enable miss-calibration
  theGainSmearing=conf_.getParameter<double>("GainSmearing"); // sigma of the gain smearing
  theOffsetSmearing=conf_.getParameter<double>("OffsetSmearing"); //sigma of the offset smearing

  //////pixel geometry
  theRowsInChip=conf_.getParameter<int>("PixelROCRows");
  theColsInChip=conf_.getParameter<int>("PixelROCCols");
  
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout<<"SiPixelDigitizerAlgorithm constructed"<<endl;
    cout<<"Configuraion parameters:"<<endl;  
    cout << "Threshold/Gain = "  
	 << thePixelThreshold << " " <<  theElectronPerADC 
	 << " " << theAdcFullScale << endl; 
    cout << " The delta cut-off is set to " << tMax << endl;
    if(doMissCalibrate) cout << " miss-calibrate the pixel amplitude " 
			     << theGainSmearing << " " << theOffsetSmearing 
			     << endl;
  }
  //MP DA RISOLVERE
  // particleTable =  &HepPDT::theTable();

}
SiPixelDigitizerAlgorithm::~SiPixelDigitizerAlgorithm(){
 if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
   cout<<"SiPixelDigitizerAlgorithm deleted"<<endl;
 }
}

//void  SiPixelDigitizerAlgorithm::run(const std::vector<PSimHit*> &input,PixelDigiCollection &output)
void  SiPixelDigitizerAlgorithm::run(const std::vector<PSimHit*> &input,PixelDigiCollection &output,PixelGeomDetUnit *pixdet)
{


  _detp = pixdet; //cache the PixGeomDetUnit

  // Pixel Efficiency moved from the constructor to the method run because
  // the information of the det are nota available in the constructor
  // Effciency parameters. 0 - no inefficiency, 1-low lumi, 10-high lumi
  // enum PixelGeomDetType::SubDetector pixelPart;

  pixelPart=pixdet->type().subDetector();


  thePixelLuminosity=conf_.getParameter<int>("AddPixelInefficiency");
  if (thePixelLuminosity>0) {
    pixelInefficiency=true;
    // Default efficiencies 
    for (int i=0; i<3;i++) {
      // Assume 1% inefficiency for single pixels, 
      // this is given by faulty bump-bonding and seus.  
      thePixelEfficiency[i]     = 1.-0.01;  // pixels = 99%
      // For columns make 1% default.
      thePixelColEfficiency[i]  = 1.-0.01;  // columns = 99%
      // A flat 0.25% inefficiency due to lost data packets from TBM
      thePixelChipEfficiency[i] = 1.-0.0025; // chips = 99.75%
    }
    
    
    
    // Special cases 

    //MP DA VERIFICARE il valore di pixelpart
//     if( pixelPart == barrel ) {  // For the barrel
    if( pixelPart == 0 ) {  // For the barrel
      if(thePixelLuminosity==10) { // For high luminosity
 	thePixelColEfficiency[0] = 1.-0.034; // 3.4% for r=4 only
	thePixelEfficiency[0]    = 1.-0.015; // 1.5% for r=4
      }        
      // For the reset assume the default.
      //} else if ( pixelPart == forward ) {  // For endcaps
      //if(thePixelLuminosity==10) { // high luminosity
      //thePixelColEfficiency[0]  = 1.-0.025; // 2.5% for disk 1, like r7
      //} else if (thePixelLuminosity==2) {
      //thePixelColEfficiency[0]  = 1.-0.008; // 0.8% for disk 1, like r7
      //} else if (thePixelLuminosity==1) {  //1*10^33 all deafult
      //}  
    }



    // Set efficencies to a preset values (Testing only),-1=not used(def)
    PixelEff=conf_.getParameter<double>("PixelEfficiency");
    PixelColEff=conf_.getParameter<double>("PixelColEfficiency");
    PixelChipEff=conf_.getParameter<double>("PixelChipEfficiency");

    if(PixelEff>0.) {     // Set all layers to the preset value
      for (int i=0; i<3;i++) {
	thePixelEfficiency[i] = PixelEfficiency;
      }
    }
    if(PixelColEff>0.) {
      for (int i=0; i<3;i++) {
	thePixelColEfficiency[i] = PixelColEfficiency;
      }
    }
    if(PixelChipEff>0.) {
      for (int i=0; i<3;i++) {
	thePixelChipEfficiency[i] = PixelChipEfficiency;
      }
    }
  }
  
  //MP QUESTA PARTE VA COMPLETAMENTE MODIFICATA NON APPENA SARANNO DISPONIBILI 
  // I SIMHIT

  unsigned int detID = 0;
  unsigned int newDetID = 0;

  int detunits = 0;
  bool first = true;
  //  vector<PixelDigi> collector;
  ss.clear();
  //raggruppati gli hit in base alla detunit
  
  
  vector<PSimHit*>::const_iterator simHitIter = input.begin();
  vector<PSimHit*>::const_iterator simHitIterEnd = input.end();
  //  vector<PseudoHit*>::const_iterator simHitIter = input.begin();
  //  vector<PseudoHit*>::const_iterator simHitIterEnd = input.end();
  //start the loop over the simhits
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
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
  
  //Digitization of the SimHits of a given pixdet
  vector<PixelDigi> collector =digitize(pixdet);
  

  //Fill the pixidigicollection
  PixelDigiCollection::Range outputRange;
  outputRange.first = collector.begin();
  outputRange.second = collector.end();
  output.put(outputRange,detID);
  collector.clear();

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << "[SiPixelDigitizerAlgorithm] converted " << digis << " StripDigis in " << detunits+1 << " DetUnits." << endl; 
  }
  
}
/**********************************************************************/

vector<PixelDigi> SiPixelDigitizerAlgorithm::digitize(PixelGeomDetUnit *det){
  

  digis = 0;
  if( ss.size() > 0 || addNoisyPixels) {
  
    topol=&det->specificTopology(); // cache topology
    numColumns = topol->ncolumns();  // det module number of cols&rows
    numRows = topol->nrows();


    moduleThickness = det->specificSurface().bounds().thickness(); // full detector thicness

    //MP DA SISTEMARE
    //     float noiseInADCCounts = _detp->readout().noiseInAdcCounts();
    float noiseInADCCounts=3.7;  
    // For the noise generation I need noise in electrons
    theNoiseInElectrons = noiseInADCCounts * theElectronPerADC;    
    // Find the threshold in electrons
    thePixelThresholdInE = thePixelThreshold * theNoiseInElectrons; 


    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ){
       cout << " PixelDigitizer " << pixelPart << " " 
	    << numColumns << " " << numRows << " " << moduleThickness<<endl;
    //MP DA SCOMMENTARE
//       cout << thePixelThreshold << " " << thePixelThresholdInE << " " 
// 	   << noiseInADCCounts << " " << theNoiseInElectrons << endl;
    }
    // produce SignalPoint's for all SimHit's in detector
    // Loop over hits
    vector<const PSimHit*>::const_iterator ssbegin = ss.begin();
    vector<const PSimHit*>::const_iterator ssend = ss.end();
    for (;ssbegin != ssend; ++ssbegin) {
      const PSimHit *pointerHit = *ssbegin;
 

      if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
	cout << pointerHit->particleType() << " " << pointerHit->pabs() << " " 
	     << pointerHit->energyLoss() << " " << pointerHit->tof() << " " 
	  //MP packedTrackId???    
	  //	  << pointerHit->packedTrackId() << " " << pointerHit->processType() << " " 
	     << pointerHit->detUnitId() << endl; 
	cout << pointerHit->entryPoint() << " " << pointerHit->exitPoint() << endl; 
      }


      _collection_points.clear();  // Clear the container
      // fill _collection_points for this SimHit, indpendent of topology
      primary_ionization(*pointerHit); // fills _ionization_points
      drift(*pointerHit);  // transforms _ionization_points to _collection_points  

      // compute induced signal on readout elements and add to _signal
      induce_signal(*pointerHit); //*ihit needed only for SimHit<-->Digi link


				      //      int adc=10;
				      //   int row=10;
				      //    int col=10; 
      //      internal_coll.push_back(PixelDigi(row,col,adc));
    }
  
    if(addNoise) add_noise();  // generate noise

    // Do only if needed 
    if(pixelInefficiency && _signal.size()>0 ) 
      pixel_inefficiency(); // Kill some pixels
  }
  make_digis();
  return internal_coll;
}


/***********************************************************************/
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
    cout << " enter primary_ionzation " << NumberOfSegments << " " ;
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
    
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
      cout << i << " " << _ionization_points[i].x() << " " 
	   << _ionization_points[i].y() << " " 
	   << _ionization_points[i].z() << " " 
	   << _ionization_points[i].energy() <<endl;
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

void SiPixelDigitizerAlgorithm::drift(const PSimHit& hit){

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {  
    cout << " enter drift " << endl;
  }
  
  _collection_points.resize( _ionization_points.size()); // set size

  LocalPoint center(0.,0.);  // detector center 

  //MP dove e' driftdirection?
  // LocalVector driftDir = _detp->driftDirection(center); // drift in center
  LocalVector driftDir(1.,1.,1.);


  if(driftDir.z() ==0.) {
    cout << " pxlx: drift in z is zero " << endl;
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



void SiPixelDigitizerAlgorithm::induce_signal( const PSimHit& hit) {

  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)
  //DA MODIFICARE
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << " enter induce_signal, " 
	 << topol->pitch().first << " " << topol->pitch().second << endl; //OK
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
       cout << " cloud " << i->position().x() << " " << i->position().y() << " " 
	    << i->sigma_x() << " " << i->sigma_y() << " " << i->amplitude() <<
	 endl;
       //    cout << CloudCenterX << " " << CloudCenterY << " " <<
       //      SigmaX << " " << SigmaY << " " << Charge << " ";
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
       cout << " right-up " << PointRightUp << " " ;
       cout << mp.x() << " " << mp.y() << " ";
       cout << IPixRightUpX << " " << IPixRightUpY << endl;
     }
 

     mp = topol->measurementPosition(PointLeftDown ); //OK
    
     int IPixLeftDownX = int( floor( mp.x()));
     int IPixLeftDownY = int( floor( mp.y()));
     
     if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) { 
       cout << " left-down " << PointLeftDown << " " ;
       cout << mp.x() << " " << mp.y() << " ";
       cout << IPixLeftDownX << " " << IPixLeftDownY << endl;
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
	 //MP da verificare
	 //	 if (status != 0) throw DetLogicError("GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen");
	 if (status != 0)  cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<endl;
	 LowerBound = 1-result.val;
	 //	cout <<" LOWERB PIXEL "<<oLowerBound<<" " <<LowerBound<<" " <<oLowerBound-LowerBound<<endl;
       }
     
       if (ix == numRows-1) UpperBound = 1.;
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
	//	cout <<" LOWERB PIXEL "<<oUpperBound<<" " <<UpperBound<<" " <<oUpperBound-UpperBound<<endl;

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
	 //MP da verificare
	 if (status != 0)  cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<endl;
	 //	 if (status != 0) throw DetLogicError("GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen");
	LowerBound = 1. - result.val;
	//	cout <<" LOWERB PIXEL "<<oLowerBound<<" " <<LowerBound<<" " <<oLowerBound-LowerBound<<endl;


      }

      if (iy == numColumns-1) UpperBound = 1.;
      else {
        mp = MeasurementPoint( 0.0, float(iy+1) );
        yUB = topol->localPosition(mp).y();
	//        float oUpperBound = freq_( (yUB-CloudCenterY)/SigmaY);
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yUB-CloudCenterY)/SigmaY, &result);
	 //MP da verificare
	 if (status != 0)  cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<endl;
	 //	 if (status != 0) throw DetLogicError("GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen");
	UpperBound = 1. - result.val;
	//	cout <<" LOWERB PIXEL "<<oUpperBound<<" " <<UpperBound<<" " <<oUpperBound-UpperBound<<endl;

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
	  cout << " pixel " << ix << " " << iy << " - ";
	  cout << chan << " " << ChargeFraction << endl; //OK
	  mp = MeasurementPoint( float(ix), float(iy) );
	  cout << mp.x() << " " << mp.y() << " "; //OK
	  LocalPoint lp = topol->localPosition(mp);
	  cout << lp.x() << " " << lp.y() << " ";  // givex edge position
	  chan = topol->channel(lp); // something wrong 1->0, 
	  cout << chan << endl; // edge belongs to previous ?
	}
	
      } // endfor iy
    } //endfor ix
   
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      
      // Test conversions
      cout << " Test " << endl;
      mp = topol->measurementPosition( i->position() ); //OK
      cout << mp.x() << " " << mp.y() << " ";
      LocalPoint lp = topol->localPosition(mp);     //OK
      cout << lp.x() << " " << lp.y() << " ";
      pair<float,float> p = topol->pixel( i->position() );  //OK
      cout << p.first << " " << p.second << " " ;
      chan = PixelDigi::pixelToChannel( int(p.first), int(p.second));
      cout << chan << " "; //OK
      pair<int,int> ip = PixelDigi::channelToPixel(chan);
      cout << ip.first << " " << ip.second << " "; //OK
      MeasurementPoint mp1 = MeasurementPoint( float(ip.first),
					       float(ip.second) );
      cout << mp1.x() << " " << mp1.y() << " "; //OK
      cout << topol->localPosition(mp1).x() << " "  //OK
	   << topol->localPosition(mp1).y() << " ";
      cout << topol->channel( i->position() ) << endl; //OK
    }
    
  } // loop over charge distributions

  // Fill the global map with all hit pixels from this event
  for ( hit_map_type::const_iterator im = hit_signal.begin();
	im != hit_signal.end(); im++) {
    _signal[(*im).first] += Amplitude( (*im).second, &hit);


    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      int chan =  (*im).first; 
      pair<int,int> ip = PixelDigi::channelToPixel(chan);
      cout << " pixel " << ip.first << " " << ip.second << " ";
      //    cout << (*im).first << " " << (*im).second << " ";    
      cout << _signal[(*im).first] << endl;    
    }

  }

}

/***********************************************************************/
//
void SiPixelDigitizerAlgorithm::make_digis() {
  internal_coll.reserve(50); internal_coll.clear();

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << " make digis ";
    cout << " pixel threshold " << thePixelThresholdInE << endl; 
    cout << " List pixels passing threshold " << endl;
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
	cout << (*i).first << " " << (*i).second << " " << signalInElectrons 
	     << " " << adc << ip.first << " " << ip.second << endl;
      }
      
      // Load digis
      internal_coll.push_back( PixelDigi( ip.first, ip.second, adc));

   
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
void SiPixelDigitizerAlgorithm::add_noise() {
  

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << " enter add_noise " << theNoiseInElectrons << endl;
  }

  // First add noise to hit pixels
  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {
    float noise  = RandGauss::shoot(0.,theNoiseInElectrons) ;
    (*i).second += Amplitude( noise,0);
    //     cout << (*i).first << " " << (*i).second << " ";
    //     cout << noise << " " << (*i).second << endl;
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
    cout <<  " Add noisy pixels " << numRows << " " << numColumns << " "
	 << theNoiseInElectrons << " " 
	 << thePixelThreshold << " " << numberOfPixels << " " 
	 << otherPixels.size() << endl;
  }

  for (mapI = otherPixels.begin(); mapI!= otherPixels.end(); mapI++) {
    int iy = ((*mapI).first) / numRows;
    int ix = ((*mapI).first) - (iy*numRows);

    // Keep for a while for testing.
    if( iy < 0 || iy > (numColumns-1) ) 
      cout << " error in iy " << iy << endl;
    if( ix < 0 || ix > (numRows-1) )
      cout << " error in ix " << ix << endl;

    int chan = PixelDigi::pixelToChannel(ix, iy);

    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      cout<<" Storing noise = " << (*mapI).first << " " << (*mapI).second 
	  << " " << ix << " " << iy << " " << chan <<endl;
    }
  

    if(_signal[chan] == 0){
      float noise = float( (*mapI).second );
      _signal[chan] = Amplitude (noise, 0);
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
  //MP da verificare
  //  if ( pixelPart == barrel ) {  // barrel layers
  if ( pixelPart == 0 ) {  // barrel layers
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
    
    pixelEfficiency  = thePixelEfficiency[layerIndex-1];
    columnEfficiency = thePixelColEfficiency[layerIndex-1];
    chipEfficiency   = thePixelChipEfficiency[layerIndex-1];
    
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
    pixelEfficiency  = thePixelEfficiency[0];
    columnEfficiency = thePixelColEfficiency[0];
    chipEfficiency   = thePixelChipEfficiency[0];

  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
    cout << " enter pixel_inefficiency " << pixelEfficiency << " " 
	 << columnEfficiency << " " << chipEfficiency << endl;
  }
  // TESTING
  //  cout << " enter pixel_inefficiency " << pixelEfficiency << " " 
  //       << columnEfficiency << " " << chipEfficiency << endl;
  //  cout << " det size = " << numColumns << " " << numRows 
  //       << " det type = " << _detp->type().part() << " " << colsInChip
  //       << " " << rowsInChip <<  endl;
  
  // Initilize the index converter
  PixelChipIndices indexConverter(theColsInChip,theRowsInChip,
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
    
    //cout <<chan << " " << pixX << " " << pixY << " " << (*i).second  << endl;
    //cout << chipX << " " << chipY << " " << row << " " << col << " " 
    // << chipIndex << " " << pixInChip << " " << dColInChip << " " 
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
  
  //cout << " pixel " << _signal.size() << endl;
  // Now loop again over pixel to kill some
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {
    
    //cout << " pix " << i->first << " " << float(i->second) << endl;    
    //    int chan = i->first;
    pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
    int pixX = ip.first + 1;  // my indices start from 1
    int pixY = ip.second + 1;
    
    indexConverter.chipIndices(pixX,pixY,chipX,chipY,row,col); //get chip index
    int chipIndex = indexConverter.chipIndex(chipX,chipY);
    
    pair<int,int> dColId = indexConverter.dColumn(row,col);  // get dcol index
    int dColInDet = indexConverter.dColumnIdInDet(dColId.second,chipIndex);

    //cout <<chan << " " << pixX << " " << pixY << " " << (*i).second  << endl;
    //cout << chipX << " " << chipY << " " << row << " " << col << " " 
    // << chipIndex << " " << dColInDet << endl;
    //cout <<  chips[chipIndex] << " " <<  columns[dColInDet] << endl; 

    float rand  = RandFlat::shoot();
    if( chips[chipIndex]==0 ||  columns[dColInDet]==0 
	|| rand>pixelEfficiency ) {
      // make pixel amplitude =0, pixel will be lost at clusterization    
      i->second.set(0.); // reset amplitude, 
      //cout << " pixel will be killed " << float(i->second) << " " 
      //   << rand << " " << chips[chipIndex] << " " 
      //   << columns[dColInDet] << endl;
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
