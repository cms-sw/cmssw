// File: DataMixingSiPixelWorker.cc
// Description:  see DataMixingSiPixelMCDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
//
//

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include "CLHEP/Random/RandFlat.h"

#include "DataMixingSiPixelMCDigiWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  //  DataMixingSiPixelMCDigiWorker::DataMixingSiPixelMCDigiWorker() { } 

  // Constructor 
  DataMixingSiPixelMCDigiWorker::DataMixingSiPixelMCDigiWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
    label_(ps.getParameter<std::string>("Label")),
    geometryType_(ps.getParameter<std::string>("GeometryType")),
    // get external parameters:
    // To account for upgrade geometries do not assume the number 
    // of layers or disks.
    NumberOfBarrelLayers(ps.exists("NumPixelBarrel")?ps.getParameter<int>("NumPixelBarrel"):3),
    NumberOfEndcapDisks(ps.exists("NumPixelEndcap")?ps.getParameter<int>("NumPixelEndcap"):2),
    theInstLumiScaleFactor(ps.getParameter<double>("theInstLumiScaleFactor")), //For dynamic inefficiency PU scaling
    bunchScaleAt25(ps.getParameter<double>("bunchScaleAt25")), //For dynamic inefficiency bunchspace scaling
    // Control the pixel inefficiency
    AddPixelInefficiency(ps.getParameter<bool>("AddPixelInefficiencyFromPython")),
    pixelEff_(ps, AddPixelInefficiency,NumberOfBarrelLayers,NumberOfEndcapDisks)
  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    pixeldigi_collectionSig_   = ps.getParameter<edm::InputTag>("pixeldigiCollectionSig");
    pixeldigi_collectionPile_   = ps.getParameter<edm::InputTag>("pixeldigiCollectionPile");
    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    PixelDigiToken_ = iC.consumes<edm::DetSetVector<PixelDigi> >(pixeldigi_collectionSig_);
    PixelDigiPToken_ = iC.consumes<edm::DetSetVector<PixelDigi> >(pixeldigi_collectionPile_);

    // clear local storage for this event                                                                     
    SiHitStorage_.clear();

  }
	       

  // Virtual destructor needed.
  DataMixingSiPixelMCDigiWorker::~DataMixingSiPixelMCDigiWorker() { 
  }  

  // Need an event initialization

  void DataMixingSiPixelMCDigiWorker::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {	

    iSetup.get<TrackerDigiGeometryRecord>().get(geometryType_, pDD); 
    //edm::ESHandle<TrackerTopology> tTopoHand;
    //iSetup.get<IdealGeometryRecord>().get(tTopoHand);
    //const TrackerTopology *tTopo=tTopoHand.product();
  }					   


  DataMixingSiPixelMCDigiWorker::PixelEfficiencies::PixelEfficiencies(const edm::ParameterSet& conf, bool AddPixelInefficiency, int NumberOfBarrelLayers, int NumberOfEndcapDisks) {
    // pixel inefficiency
    // Don't use Hard coded values, read inefficiencies in from python or don't use any
    int NumberOfTotLayers = NumberOfBarrelLayers + NumberOfEndcapDisks;
    FPixIndex=NumberOfBarrelLayers;
    if (AddPixelInefficiency){
      int i=0;
      thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix1");
      thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix2");
      thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix3");
      if (NumberOfBarrelLayers>=4){thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix4");}
      //
      i=0;
      thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix1");
      thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix2");
      thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix3");
      if (NumberOfBarrelLayers>=4){thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix4");}
      //
      i=0;
      thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix1");
      thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix2");
      thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix3");
      if (NumberOfBarrelLayers>=4){thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix4");}
      //
      i=0;
      theLadderEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theLadderEfficiency_BPix1");
      theLadderEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theLadderEfficiency_BPix2");
      theLadderEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theLadderEfficiency_BPix3");
      if ( ((theLadderEfficiency_BPix[0].size()!=20) || (theLadderEfficiency_BPix[1].size()!=32) ||
	    (theLadderEfficiency_BPix[2].size()!=44)) && (NumberOfBarrelLayers==3) )  
	throw cms::Exception("Configuration") << "Wrong ladder number in efficiency config!";
      //		     
      i=0;
      theModuleEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theModuleEfficiency_BPix1");
      theModuleEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theModuleEfficiency_BPix2");
      theModuleEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theModuleEfficiency_BPix3");
      if ( ((theModuleEfficiency_BPix[0].size()!=4) || (theModuleEfficiency_BPix[1].size()!=4) ||
	    (theModuleEfficiency_BPix[2].size()!=4)) && (NumberOfBarrelLayers==3) )  
	throw cms::Exception("Configuration") << "Wrong module number in efficiency config!";
      //
      i=0;		     
      thePUEfficiency[i++] = conf.getParameter<std::vector<double> >("thePUEfficiency_BPix1");
      thePUEfficiency[i++] = conf.getParameter<std::vector<double> >("thePUEfficiency_BPix2");
      thePUEfficiency[i++] = conf.getParameter<std::vector<double> >("thePUEfficiency_BPix3");		    		    
      if ( ((thePUEfficiency[0].size()==0) || (thePUEfficiency[1].size()==0) || 
	    (thePUEfficiency[2].size()==0)) && (NumberOfBarrelLayers==3) )
	throw cms::Exception("Configuration") << "At least one PU efficiency (BPix) number is needed in efficiency config!";
      // The next is needed for Phase2 Tracker studies
      if (NumberOfBarrelLayers>=5){
	if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
	// For Phase2 tracker layers just set the outermost BPix inefficiency to 99.9% THESE VALUES ARE HARDCODED ALSO ELSEWHERE IN THIS FILE
	for (int j=5 ; j<=NumberOfBarrelLayers ; j++){
	  thePixelColEfficiency[j-1]=0.999;
	  thePixelEfficiency[j-1]=0.999;
	  thePixelChipEfficiency[j-1]=0.999;
	}
      }
      //
      i=FPixIndex;
      thePixelColEfficiency[i++]   = conf.getParameter<double>("thePixelColEfficiency_FPix1");
      thePixelColEfficiency[i++]   = conf.getParameter<double>("thePixelColEfficiency_FPix2");
      if (NumberOfEndcapDisks>=3){thePixelColEfficiency[i++]   = conf.getParameter<double>("thePixelColEfficiency_FPix3");}
      i=FPixIndex;
      thePixelEfficiency[i++]      = conf.getParameter<double>("thePixelEfficiency_FPix1");
      thePixelEfficiency[i++]      = conf.getParameter<double>("thePixelEfficiency_FPix2");
      if (NumberOfEndcapDisks>=3){thePixelEfficiency[i++]      = conf.getParameter<double>("thePixelEfficiency_FPix3");}
      i=FPixIndex;
      thePixelChipEfficiency[i++]  = conf.getParameter<double>("thePixelChipEfficiency_FPix1");
      thePixelChipEfficiency[i++]  = conf.getParameter<double>("thePixelChipEfficiency_FPix2");
      if (NumberOfEndcapDisks>=3){thePixelChipEfficiency[i++]  = conf.getParameter<double>("thePixelChipEfficiency_FPix3");}
      // The next is needed for Phase2 Tracker studies
      if (NumberOfEndcapDisks>=4){
	if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
	// For Phase2 tracker layers just set the extra FPix disk inefficiency to 99.9% THESE VALUES ARE HARDCODED ALSO ELSEWHERE IN THIS FILE
	for (int j=4+FPixIndex ; j<=NumberOfEndcapDisks+NumberOfBarrelLayers ; j++){
	  thePixelColEfficiency[j-1]=0.999;
	  thePixelEfficiency[j-1]=0.999;
	  thePixelChipEfficiency[j-1]=0.999;
	}
      }
      //FPix Dynamic Inefficiency
      i=FPixIndex;
      theInnerEfficiency_FPix[i++] = conf.getParameter<double>("theInnerEfficiency_FPix1");
      theInnerEfficiency_FPix[i++] = conf.getParameter<double>("theInnerEfficiency_FPix2");
      i=FPixIndex;
      theOuterEfficiency_FPix[i++] = conf.getParameter<double>("theOuterEfficiency_FPix1");
      theOuterEfficiency_FPix[i++] = conf.getParameter<double>("theOuterEfficiency_FPix2");
      i=FPixIndex;
      thePUEfficiency[i++] = conf.getParameter<std::vector<double> >("thePUEfficiency_FPix_Inner");
      thePUEfficiency[i++] = conf.getParameter<std::vector<double> >("thePUEfficiency_FPix_Outer");
      if ( ((thePUEfficiency[3].size()==0) || (thePUEfficiency[4].size()==0)) && (NumberOfEndcapDisks==2) )
	throw cms::Exception("Configuration") << "At least one (FPix) PU efficiency number is needed in efficiency config!";
    }
    // the first "NumberOfBarrelLayers" settings [0],[1], ... , [NumberOfBarrelLayers-1] are for the barrel pixels
    // the next  "NumberOfEndcapDisks"  settings [NumberOfBarrelLayers],[NumberOfBarrelLayers+1], ... [NumberOfEndcapDisks+NumberOfBarrelLayers-1]
  }




  void DataMixingSiPixelMCDigiWorker::addSiPixelSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingSiPixelMCDigiWorker")<<"===============> adding MC signals for "<<e.id();

    Handle< edm::DetSetVector<PixelDigi> >  input;

    if( e.getByToken(PixelDigiToken_,input) ) {

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<PixelDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiPixelMCDigiWorker")  << "Processing DetID " << DSViter->id;
#endif

	uint32_t detID = DSViter->id;
	edm::DetSet<PixelDigi>::const_iterator begin =(DSViter->data).begin();
	edm::DetSet<PixelDigi>::const_iterator end   =(DSViter->data).end();
	edm::DetSet<PixelDigi>::const_iterator icopy;
  
	OneDetectorMap LocalMap;

	for (icopy=begin; icopy!=end; icopy++) {
	  LocalMap.insert(OneDetectorMap::value_type( (icopy->channel()), *icopy ));
	}

	SiHitStorage_.insert( SiGlobalIndex::value_type( detID, LocalMap ) );
      }
 
    }    
  } // end of addSiPixelSignals



  void DataMixingSiPixelMCDigiWorker::addSiPixelPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                  ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingSiPixelMCDigiWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    std::shared_ptr<Wrapper<edm::DetSetVector<PixelDigi> >  const> inputPTR =
      getProductByTag<edm::DetSetVector<PixelDigi> >(*ep, pixeldigi_collectionPile_, mcc);

    if(inputPTR ) {

      const edm::DetSetVector<PixelDigi>  *input = const_cast< edm::DetSetVector<PixelDigi> * >(inputPTR->product());



      //   Handle< edm::DetSetVector<PixelDigi> >  input;

      //   if( e->getByLabel(pixeldigi_collectionPile_,input) ) {

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<PixelDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiPixelMCDigiWorker")  << "Pileups: Processing DetID " << DSViter->id;
#endif

	uint32_t detID = DSViter->id;
	edm::DetSet<PixelDigi>::const_iterator begin =(DSViter->data).begin();
	edm::DetSet<PixelDigi>::const_iterator end   =(DSViter->data).end();
	edm::DetSet<PixelDigi>::const_iterator icopy;

	// find correct local map (or new one) for this detector ID

	SiGlobalIndex::const_iterator itest;

	itest = SiHitStorage_.find(detID);

	if(itest!=SiHitStorage_.end()) {  // this detID already has hits, add to existing map

	  OneDetectorMap LocalMap = itest->second;

	  // fill in local map with extra channels
	  for (icopy=begin; icopy!=end; icopy++) {
	    LocalMap.insert(OneDetectorMap::value_type( (icopy->channel()), *icopy ));
	  }

	  SiHitStorage_[detID]=LocalMap;
	  
	}
	else{ // fill local storage with this information, put in global collection

	  OneDetectorMap LocalMap;

	  for (icopy=begin; icopy!=end; icopy++) {
	    LocalMap.insert(OneDetectorMap::value_type( (icopy->channel()), *icopy ));
	  }

	  SiHitStorage_.insert( SiGlobalIndex::value_type( detID, LocalMap ) );
	}

      }
    }
  }


 
  void DataMixingSiPixelMCDigiWorker::putSiPixel(edm::Event &e, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> &ps, int &bs) {

    // collection of Digis to put in the event

    std::vector< edm::DetSet<PixelDigi> > vPixelDigi;

    // loop through our collection of detectors, merging hits and putting new ones in the output

    _signal.clear();

    // big loop over Detector IDs:

    for(SiGlobalIndex::const_iterator IDet = SiHitStorage_.begin();
	IDet != SiHitStorage_.end(); IDet++) {

      uint32_t detID = IDet->first;      

      OneDetectorMap LocalMap = IDet->second;

      signal_map_type Signals;
      Signals.clear();

      //counter variables
      int formerPixel = -1;
      int currentPixel;
      int ADCSum = 0;


      OneDetectorMap::const_iterator iLocalchk;

      for(OneDetectorMap::const_iterator iLocal  = LocalMap.begin();
	  iLocal != LocalMap.end(); ++iLocal) {

	currentPixel = iLocal->first; 

	if (currentPixel == formerPixel) { // we have to add these digis together
	  ADCSum+=(iLocal->second).adc();
	}
	else{
	  if(formerPixel!=-1){             // ADC info stolen from SiStrips...
	    if (ADCSum > 511) ADCSum = 255;
	    else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;

	    Signals.insert( std::make_pair(formerPixel, ADCSum));
	    //PixelDigi aHit(formerPixel, ADCSum);
	    //SPD.push_back( aHit );	  
	  }
	  // save pointers for next iteration
	  formerPixel = currentPixel;
	  ADCSum = (iLocal->second).adc();
	}

	iLocalchk = iLocal;
	if((++iLocalchk) == LocalMap.end()) {  //make sure not to lose the last one
	  if (ADCSum > 511) ADCSum = 255;
	  else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	  Signals.insert( std::make_pair(formerPixel, ADCSum));
	  //SPD.push_back( PixelDigi(formerPixel, ADCSum) );	  
	} 

      }// end of loop over one detector

      // stick this into the global vector of detector info
      _signal.insert( std::make_pair( detID, Signals));

    } // end of big loop over all detector IDs

    // put the collection of digis in the event   
    LogInfo("DataMixingSiPixelMCDigiWorker") << "total # Merged Pixels: " << _signal.size() ;

    // Now, we have to run Lumi-Dependent efficiency calculation on the merged pixels.
    // This is the only place where we have the PreMixed pileup information so that we can calculate
    // the instantaneous luminosity and do the dynamic inefficiency.

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<IdealGeometryRecord>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    // set pileup information.

    setPileupInfo(ps, bs);

    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
      
      if((*iu)->type().isTrackerPixel()) {

	//
	const PixelGeomDetUnit* pixdet = dynamic_cast<const PixelGeomDetUnit*>((*iu));
	uint32_t detID = pixdet->geographicalId().rawId();

	// fetch merged hits for this detID

	signal_map_type& theSignal = _signal[detID];

	// if we have some hits...
	if(theSignal.size()>0) {

	  edm::DetSet<PixelDigi> SPD(detID);  // make empty vector with this detID so we can push back digis at the end 

	  const PixelTopology* topol=&pixdet->specificTopology();
	  int numColumns = topol->ncolumns();  // det module number of cols&rows
	  int numRows = topol->nrows();

	  // do inefficiency calculation, drop some pixel hits

	  // Predefined efficiencies
	  double pixelEfficiency  = 1.0;
	  double columnEfficiency = 1.0;
	  double chipEfficiency   = 1.0;
  
	  // setup the chip indices conversion
	  if    (pixdet->subDetector()==GeomDetEnumerators::SubDetector::PixelBarrel ||
		 pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXB){// barrel layers
	    int layerIndex=tTopo->layer(detID);
	    pixelEfficiency  = pixelEff_.thePixelEfficiency[layerIndex-1];
	    columnEfficiency = pixelEff_.thePixelColEfficiency[layerIndex-1];
	    chipEfficiency   = pixelEff_.thePixelChipEfficiency[layerIndex-1];
	    //std::cout <<"Using BPix columnEfficiency = "<<columnEfficiency<< " for layer = "<<layerIndex <<"\n";
	    // This should never happen, but only check if it is not an upgrade geometry
	    if (NumberOfBarrelLayers==3){
	      if(numColumns>416)  LogWarning ("Pixel Geometry") <<" wrong columns in barrel "<<numColumns;
	      if(numRows>160)  LogWarning ("Pixel Geometry") <<" wrong rows in barrel "<<numRows;
      
	      int ladder=tTopo->pxbLadder(detID);
	      int module=tTopo->pxbModule(detID);
	      if (module<=4) module=5-module;
	      else module-=4;
      
	      columnEfficiency *= pixelEff_.theLadderEfficiency_BPix[layerIndex-1][ladder-1]*pixelEff_.theModuleEfficiency_BPix[layerIndex-1][module-1]*_pu_scale[layerIndex-1];
	    }
	  } else if(pixdet->subDetector()==GeomDetEnumerators::SubDetector::PixelEndcap ||
		    pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXEC ||
		    pixdet->subDetector()==GeomDetEnumerators::SubDetector::P2PXEC){                // forward disks

	    unsigned int diskIndex=tTopo->layer(detID)+pixelEff_.FPixIndex; // Use diskIndex-1 later to stay consistent with BPix
	    unsigned int panelIndex=tTopo->pxfPanel(detID);
	    unsigned int moduleIndex=tTopo->pxfModule(detID);
	    //if (pixelEff_.FPixIndex>diskIndex-1){throw cms::Exception("Configuration") <<"SiPixelDigitizer is using the wrong efficiency value. index = "
	    //                                                                       <<diskIndex-1<<" , MinIndex = "<<pixelEff_.FPixIndex<<" ... "<<tTopo->pxfDisk(detID);}
	    pixelEfficiency  = pixelEff_.thePixelEfficiency[diskIndex-1];
	    columnEfficiency = pixelEff_.thePixelColEfficiency[diskIndex-1];
	    chipEfficiency   = pixelEff_.thePixelChipEfficiency[diskIndex-1];
	    //std::cout <<"Using FPix columnEfficiency = "<<columnEfficiency<<" for Disk = "<< tTopo->pxfDisk(detID)<<"\n";
	    // Sometimes the forward pixels have wrong size,
	    // this crashes the index conversion, so exit, but only check if it is not an upgrade geometry
	    if (NumberOfBarrelLayers==3){  // whether it is the present or the phase 1 detector can be checked using GeomDetEnumerators::SubDetector
	      if(numColumns>260 || numRows>160) {
		if(numColumns>260)  LogWarning ("Pixel Geometry") <<" wrong columns in endcaps "<<numColumns;
		if(numRows>160)  LogWarning ("Pixel Geometry") <<" wrong rows in endcaps "<<numRows;
		return;
	      }
	      if ((panelIndex==1 && (moduleIndex==1 || moduleIndex==2)) || (panelIndex==2 && moduleIndex==1)) { //inner modules
		columnEfficiency*=pixelEff_.theInnerEfficiency_FPix[diskIndex-1]*_pu_scale[3];
	      } else { //outer modules
		columnEfficiency*=pixelEff_.theOuterEfficiency_FPix[diskIndex-1]*_pu_scale[4];
	      }
	    } // current detector, forward
	  } else if(pixdet->subDetector()==GeomDetEnumerators::SubDetector::P2OTB ||pixdet->subDetector()==GeomDetEnumerators::SubDetector::P2OTEC) {
	    // If phase 2 outer tracker, hardcoded values as they have been so far
	    pixelEfficiency  = 0.999;
	    columnEfficiency = 0.999;
	    chipEfficiency   = 0.999;
	  } // if barrel/forward
  

	  // Initilize the index converter
	  //PixelIndices indexConverter(numColumns,numRows);
	  std::auto_ptr<PixelIndices> pIndexConverter(new PixelIndices(numColumns,numRows));

	  int chipIndex = 0;
	  int rowROC = 0;
	  int colROC = 0;
	  std::map<int, int, std::less<int> >chips, columns;
	  std::map<int, int, std::less<int> >::iterator iter;
  
	  // Find out the number of columns and rocs hits
	  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
	  for (signal_map_const_iterator i = theSignal.begin(); i != theSignal.end(); ++i) {
    
	    int chan = i->first;
	    std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
	    int row = ip.first;  // X in row
	    int col = ip.second; // Y is in col
	    //transform to ROC index coordinates
	    pIndexConverter->transformToROC(col,row,chipIndex,colROC,rowROC);
	    int dColInChip = pIndexConverter->DColumn(colROC); // get ROC dcol from ROC col
	    //dcol in mod
	    int dColInDet = pIndexConverter->DColumnInModule(dColInChip,chipIndex);
    
	    chips[chipIndex]++;
	    columns[dColInDet]++;
	  }
  
	  // Delete some ROC hits.
	  for ( iter = chips.begin(); iter != chips.end() ; iter++ ) {
	    //float rand  = RandFlat::shoot();
	    float rand  = CLHEP::RandFlat::shoot(engine);
	    if( rand > chipEfficiency ) chips[iter->first]=0;
	  }
  
	  // Delete some Dcol hits.
	  for ( iter = columns.begin(); iter != columns.end() ; iter++ ) {
	    //float rand  = RandFlat::shoot();
	    float rand  = CLHEP::RandFlat::shoot(engine);
	    if( rand > columnEfficiency ) columns[iter->first]=0;
	  }
  
	  // Now loop again over pixels to kill some of them.
	  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
	  for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {
    
	    //    int chan = i->first;
	    std::pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
	    int row = ip.first;  // X in row
	    int col = ip.second; // Y is in col
	    //transform to ROC index coordinates
	    pIndexConverter->transformToROC(col,row,chipIndex,colROC,rowROC);
	    int dColInChip = pIndexConverter->DColumn(colROC); //get ROC dcol from ROC col
	    //dcol in mod
	    int dColInDet = pIndexConverter->DColumnInModule(dColInChip,chipIndex);
    
	    //float rand  = RandFlat::shoot();
	    float rand  = CLHEP::RandFlat::shoot(engine);
	    if( chips[chipIndex]==0 || columns[dColInDet]==0
		|| rand>pixelEfficiency ) {
	      // make pixel amplitude =0, pixel will be lost at clusterization
	      i->second=(0.); // reset amplitude
	    } // end if
	    //Make a new Digi:

	    SPD.push_back( PixelDigi(i->first, i->second) );     

	  } // end pixel loop
	  // push back vector here of one detID
	  
	  vPixelDigi.push_back(SPD);
	}
      }
    }// end of loop over detectors

    // make new digi collection
    
    std::auto_ptr< edm::DetSetVector<PixelDigi> > MyPixelDigis(new edm::DetSetVector<PixelDigi>(vPixelDigi) );

    // put collection

    e.put( MyPixelDigis, PixelDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
  }

void DataMixingSiPixelMCDigiWorker::setPileupInfo(const std::vector<PileupSummaryInfo> &ps, const int &bunchSpacing) {

  double bunchScale=1.0;
  if (bunchSpacing==25) bunchScale=bunchScaleAt25;
  
  int p = -1;
  for ( unsigned int i=0; i<ps.size(); i++) 
    if ( ps[i].getBunchCrossing() == 0 ) 
      p=i;
    
  if ( p>=0 ) {
    for (size_t i=0; i<5; i++) {
      double instlumi = ps[p].getTrueNumInteractions()*theInstLumiScaleFactor*bunchScale;
      double instlumi_pow=1.;
      _pu_scale[i] = 0;
      for  (size_t j=0; j<pixelEff_.thePUEfficiency[i].size(); j++){
	_pu_scale[i]+=instlumi_pow*pixelEff_.thePUEfficiency[i][j];
	instlumi_pow*=instlumi;
      }
    }
  }


} //this sets pu_scale

} //edm
