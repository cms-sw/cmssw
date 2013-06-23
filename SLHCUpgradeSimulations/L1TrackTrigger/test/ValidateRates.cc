/////////////////////////////
// Track Trigger Checklist //
// Rates                   //
//                         //
// Nicola Pozzobon - 2013  //
// Sebastien Viret         //
/////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
//#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
//#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
//#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1D.h>
#include <TH2D.h>

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class ValidateRates : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit ValidateRates(const edm::ParameterSet& iConfig);
    virtual ~ValidateRates();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:
    /// Map to find the StackedTrackerDetId from the DetId
    std::map< DetId, StackedTrackerDetId > theBigFatMap;

    /// Event counter
    TH1D* hEventCounter;

    /// Rate in the Barrel
    std::map< unsigned int, TH2D* > mapLayer_hSimHitRate_All;
    std::map< unsigned int, TH2D* > mapLayer_hPixelDigiRate_All;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkClusterRate_All;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkStubRate_All;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkClusterRate_Good;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkStubRate_Good;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkClusterRate_Comb;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkStubRate_Comb;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkClusterRate_Unkn;
    std::map< unsigned int, TH2D* > mapLayer_hL1TkStubRate_Unkn;

    /// Rate in the Endcap
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hSimHitRate_All;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hPixelDigiRate_All;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkClusterRate_All;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkStubRate_All;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkClusterRate_Good;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkStubRate_Good;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkClusterRate_Comb;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkStubRate_Comb;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkClusterRate_Unkn;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapDisk_hL1TkStubRate_Unkn;

};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
ValidateRates::ValidateRates(edm::ParameterSet const& iConfig) 
{
  /// Insert here what you need to initialize
}

/////////////
// DESTRUCTOR
ValidateRates::~ValidateRates()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void ValidateRates::endJob()
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " ValidateRates::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void ValidateRates::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " ValidateRates::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service<TFileService> fs;

  hEventCounter = fs->make<TH1D>( "hEventCounter", "Event Counter", 1, -0.5, 0.5 );
  hEventCounter->Sumw2();

  for ( unsigned int stack = 0; stack < 12; stack++ )
  {
    histoName.str("");    histoTitle.str("");
    histoName << "hSimHitRate_All_L" << stack;
    histoTitle << "Rate of SimHits, L"<< stack;
    mapLayer_hSimHitRate_All[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hPixelDigiRate_All_L" << stack;
    histoTitle << "Rate of PixelDigis, L"<< stack;
    mapLayer_hPixelDigiRate_All[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                          150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkClusterRate_All_L" << stack;
    histoTitle << "Rate of L1TkClusters, L"<< stack;
    mapLayer_hL1TkClusterRate_All[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkStubRate_All_L" << stack;
    histoTitle << "Rate of L1TkStubs, L"<< stack;
    mapLayer_hL1TkStubRate_All[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkClusterRate_Good_L" << stack;
    histoTitle << "Rate of Good L1TkClusters, L"<< stack;
    mapLayer_hL1TkClusterRate_Good[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkStubRate_Good_L" << stack;
    histoTitle << "Rate of Good L1TkStubs, L"<< stack;
    mapLayer_hL1TkStubRate_Good[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                          150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkClusterRate_Comb_L" << stack;
    histoTitle << "Rate of Comb L1TkClusters, L"<< stack;
    mapLayer_hL1TkClusterRate_Comb[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkStubRate_Comb_L" << stack;
    histoTitle << "Rate of Comb L1TkStubs, L"<< stack;
    mapLayer_hL1TkStubRate_Comb[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                          150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkClusterRate_Unkn_L" << stack;
    histoTitle << "Rate of Unkn L1TkClusters, L"<< stack;
    mapLayer_hL1TkClusterRate_Unkn[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             150, -0.5, 149.5, 150, -0.5, 149.5 );

    histoName.str("");    histoTitle.str("");
    histoName << "hL1TkStubRate_Unkn_L" << stack;
    histoTitle << "Rate of Unkn L1TkStubs, L"<< stack;
    mapLayer_hL1TkStubRate_Unkn[ stack ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                          150, -0.5, 149.5, 150, -0.5, 149.5 );

    mapLayer_hSimHitRate_All[ stack ]->Sumw2();
    mapLayer_hPixelDigiRate_All[ stack ]->Sumw2();
    mapLayer_hL1TkClusterRate_All[ stack ]->Sumw2();
    mapLayer_hL1TkStubRate_All[ stack ]->Sumw2();
    mapLayer_hL1TkClusterRate_Good[ stack ]->Sumw2();
    mapLayer_hL1TkStubRate_Good[ stack ]->Sumw2();
    mapLayer_hL1TkClusterRate_Comb[ stack ]->Sumw2();
    mapLayer_hL1TkStubRate_Comb[ stack ]->Sumw2();
    mapLayer_hL1TkClusterRate_Unkn[ stack ]->Sumw2();
    mapLayer_hL1TkStubRate_Unkn[ stack ]->Sumw2();

    for ( unsigned int side = 0; side < 3; side++ )
    {
      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( side, stack );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hSimHitRate_All_D" << stack << "_S" << side;
      histoTitle << "Rate of SimHits, D"<< stack << "_S" << side;
      mapDisk_hSimHitRate_All[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         150, -0.5, 149.5, 150, -0.5, 149.5 );

      histoName.str("");    histoTitle.str("");
      histoName << "hPixelDigiRate_All_D" << stack << "_S" << side;
      histoTitle << "Rate of PixelDigis, D"<< stack << "_S" << side;
      mapDisk_hPixelDigiRate_All[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkClusterRate_All_D" << stack << "_S" << side;
      histoTitle << "Rate of L1TkClusters, D"<< stack << "_S" << side;
      mapDisk_hL1TkClusterRate_All[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                              150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkStubRate_All_D" << stack << "_S" << side;
      histoTitle << "Rate of L1TkStubs, D"<< stack << "_S" << side;
      mapDisk_hL1TkStubRate_All[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                           150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkClusterRate_Good_D" << stack << "_S" << side;
      histoTitle << "Rate of Good L1TkClusters, D"<< stack << "_S" << side;
      mapDisk_hL1TkClusterRate_Good[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkStubRate_Good_D" << stack << "_S" << side;
      histoTitle << "Rate of Good L1TkStubs, D"<< stack << "_S" << side;
      mapDisk_hL1TkStubRate_Good[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkClusterRate_Comb_D" << stack << "_S" << side;
      histoTitle << "Rate of Comb L1TkClusters, D"<< stack << "_S" << side;
      mapDisk_hL1TkClusterRate_Comb[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkStubRate_Comb_D" << stack << "_S" << side;
      histoTitle << "Rate of Comb L1TkStubs, D"<< stack << "_S" << side;
      mapDisk_hL1TkStubRate_Comb[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkClusterRate_Unkn_D" << stack << "_S" << side;
      histoTitle << "Rate of Unkn L1TkClusters, D"<< stack << "_S" << side;
      mapDisk_hL1TkClusterRate_Unkn[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      histoName.str("");    histoTitle.str("");
      histoName << "hL1TkStubRate_Unkn_D" << stack << "_S" << side;
      histoTitle << "Rate of Unkn L1TkStubs, D"<< stack << "_S" << side;
      mapDisk_hL1TkStubRate_Unkn[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            150, -0.5, 149.5, 150, -0.5, 149.5 );
  
      mapDisk_hSimHitRate_All[ mapKey ]->Sumw2();
      mapDisk_hPixelDigiRate_All[ mapKey ]->Sumw2();
      mapDisk_hL1TkClusterRate_All[ mapKey ]->Sumw2();
      mapDisk_hL1TkStubRate_All[ mapKey ]->Sumw2();
      mapDisk_hL1TkClusterRate_Good[ mapKey ]->Sumw2();
      mapDisk_hL1TkStubRate_Good[ mapKey ]->Sumw2();
      mapDisk_hL1TkClusterRate_Comb[ mapKey ]->Sumw2();
      mapDisk_hL1TkStubRate_Comb[ mapKey ]->Sumw2();
      mapDisk_hL1TkClusterRate_Unkn[ mapKey ]->Sumw2();
      mapDisk_hL1TkStubRate_Unkn[ mapKey ]->Sumw2();
    }
  }

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void ValidateRates::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// First of all, update the event counter
  hEventCounter->Fill(0);

  /// Geometry handles etc
//  edm::ESHandle< TrackerGeometry >                geometryHandle;
//  const TrackerGeometry*                          theGeometry;
  edm::ESHandle< StackedTrackerGeometry >         stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

  /// Geometry setup
  /// Set pointers to Geometry
//  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
//  theGeometry = &(*geometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product(); /// Note this is different 
                                                        /// from the "global" geometry

  /// Fill the map once for all
  if ( hEventCounter->GetEntries() == 1 )
  {
    /// Loop over the detector elements
    for ( StackedTrackerIterator = theStackedGeometry->stacks().begin();
          StackedTrackerIterator != theStackedGeometry->stacks().end();
          ++StackedTrackerIterator )
    {
      StackedTrackerDetId stackDetId = (*StackedTrackerIterator)->Id();
      assert( (*StackedTrackerIterator) == theStackedGeometry->idToStack(stackDetId));

      /// GeomDet and GeomDetUnit are needed to access each
      /// DetId and topology and geometric features
      /// Convert to specific DetId
      const GeomDet* det0 = theStackedGeometry->idToDet(stackDetId, 0);
      //const GeomDet* det1 = theStackedGeometry->idToDet(stackDetId, 1);

      DetId detId0 = det0->geographicalId();
      //DetId detId1 = det1->geographicalId();

      theBigFatMap.insert( std::make_pair( detId0, stackDetId ) );
      //theBigFatMap.insert( std::make_pair( detId1, stackDetId ) );

      /// NOTE
      /// Since for SimHit rates and for PixelDigi Rates it is much easier to
      /// provide them for the inner sensor only (to keep the same module numbering
      /// as for the Pt modules, otherwise there would be no reason to build
      /// theBigFatMap) theBigFatMap maps only inner sensors
    }
  }

  //////////////////
  // GET SIM HITS //
  //////////////////
  edm::Handle<edm::PSimHitContainer> simHitHandlePXBH;
  edm::Handle<edm::PSimHitContainer> simHitHandlePXBL;
  edm::Handle<edm::PSimHitContainer> simHitHandlePXFH;
  edm::Handle<edm::PSimHitContainer> simHitHandlePXFL;
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelBarrelHighTof", simHitHandlePXBH);
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelBarrelLowTof",  simHitHandlePXBL);
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelEndcapHighTof", simHitHandlePXFH);
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelEndcapLowTof",  simHitHandlePXFL);

  /// Loop over Sim Hits collections
  edm::PSimHitContainer::const_iterator iterSimHit;
  for ( iterSimHit = simHitHandlePXBH->begin();
        iterSimHit != simHitHandlePXBH->end();
        iterSimHit++ )
  {
    /// Find the StackedTrackerDetId
    if ( theBigFatMap.find( iterSimHit->detUnitId() ) == theBigFatMap.end() )
      continue;

    StackedTrackerDetId stackDetId = theBigFatMap.find( iterSimHit->detUnitId() )->second;

    /// Barrel
    if ( stackDetId.isBarrel() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = stackDetId.iLayer();
      uint32_t iPhi   = stackDetId.iPhi();
      uint32_t iZ     = stackDetId.iZ();
      mapLayer_hSimHitRate_All[ iStack ]->Fill( iZ, iPhi );
    }
    /// Endcap
    else if ( stackDetId.isEndcap() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = stackDetId.iSide();
      uint32_t iStack = stackDetId.iDisk();
      uint32_t iRing  = stackDetId.iRing();
      uint32_t iPhi   = stackDetId.iPhi();

      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
      mapDisk_hSimHitRate_All[ mapKey ]->Fill( iRing, iPhi );
    }
  }
  for ( iterSimHit = simHitHandlePXBL->begin();
        iterSimHit != simHitHandlePXBL->end();
        iterSimHit++ )
  {
    /// Find the StackedTrackerDetId
    if ( theBigFatMap.find( iterSimHit->detUnitId() ) == theBigFatMap.end() )
      continue;

    StackedTrackerDetId stackDetId = theBigFatMap.find( iterSimHit->detUnitId() )->second;

    /// Barrel
    if ( stackDetId.isBarrel() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = stackDetId.iLayer();
      uint32_t iPhi   = stackDetId.iPhi();
      uint32_t iZ     = stackDetId.iZ();
      mapLayer_hSimHitRate_All[ iStack ]->Fill( iZ, iPhi );
    }
    /// Endcap
    else if ( stackDetId.isEndcap() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = stackDetId.iSide();
      uint32_t iStack = stackDetId.iDisk();
      uint32_t iRing  = stackDetId.iRing();
      uint32_t iPhi   = stackDetId.iPhi();

      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
      mapDisk_hSimHitRate_All[ mapKey ]->Fill( iRing, iPhi );
    }
  }
  for ( iterSimHit = simHitHandlePXFH->begin();
        iterSimHit != simHitHandlePXFH->end();
        iterSimHit++ )
  {
    /// Find the StackedTrackerDetId
    if ( theBigFatMap.find( iterSimHit->detUnitId() ) == theBigFatMap.end() )
      continue;

    StackedTrackerDetId stackDetId = theBigFatMap.find( iterSimHit->detUnitId() )->second;

    /// Barrel
    if ( stackDetId.isBarrel() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = stackDetId.iLayer();
      uint32_t iPhi   = stackDetId.iPhi();
      uint32_t iZ     = stackDetId.iZ();
      mapLayer_hSimHitRate_All[ iStack ]->Fill( iZ, iPhi );
    }
    /// Endcap
    else if ( stackDetId.isEndcap() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = stackDetId.iSide();
      uint32_t iStack = stackDetId.iDisk();
      uint32_t iRing  = stackDetId.iRing();
      uint32_t iPhi   = stackDetId.iPhi();

      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
      mapDisk_hSimHitRate_All[ mapKey ]->Fill( iRing, iPhi );
    }
  }
  for ( iterSimHit = simHitHandlePXFL->begin();
        iterSimHit != simHitHandlePXFL->end();
        iterSimHit++ )
  {
    /// Find the StackedTrackerDetId
    if ( theBigFatMap.find( iterSimHit->detUnitId() ) == theBigFatMap.end() )
      continue;

    StackedTrackerDetId stackDetId = theBigFatMap.find( iterSimHit->detUnitId() )->second;

    /// Barrel
    if ( stackDetId.isBarrel() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = stackDetId.iLayer();
      uint32_t iPhi   = stackDetId.iPhi();
      uint32_t iZ     = stackDetId.iZ();
      mapLayer_hSimHitRate_All[ iStack ]->Fill( iZ, iPhi );
    }
    /// Endcap
    else if ( stackDetId.isEndcap() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = stackDetId.iSide();
      uint32_t iStack = stackDetId.iDisk();
      uint32_t iRing  = stackDetId.iRing();
      uint32_t iPhi   = stackDetId.iPhi();

      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
      mapDisk_hSimHitRate_All[ mapKey ]->Fill( iRing, iPhi );
    }
  }

  /////////////////////
  // GET PIXEL DIGIS //
  /////////////////////
  edm::Handle< edm::DetSetVector< PixelDigi > >         PixelDigiHandle;
//  edm::Handle< edm::DetSetVector< PixelDigiSimLink > >  PixelDigiSimLinkHandle;
  iEvent.getByLabel( "simSiPixelDigis", PixelDigiHandle );
//  iEvent.getByLabel( "simSiPixelDigis", PixelDigiSimLinkHandle );

  edm::DetSetVector<PixelDigi>::const_iterator detsIter;
  edm::DetSet<PixelDigi>::const_iterator       hitsIter;

  /// Loop over detector elements identifying PixelDigis
  for ( detsIter = PixelDigiHandle->begin();
        detsIter != PixelDigiHandle->end();
        detsIter++ )
  {
    DetId tkId = detsIter->id;

    /// Find the StackedTrackerDetId
    if ( theBigFatMap.find( tkId ) == theBigFatMap.end() )
      continue;

    StackedTrackerDetId stackDetId = theBigFatMap.find( tkId )->second;

    /// Loop over Digis in this specific detector element
    for ( hitsIter = detsIter->data.begin();
          hitsIter != detsIter->data.end();
          hitsIter++ )
    {
      /// Threshold (here it is NOT redundant)
      if ( hitsIter->adc() <= 30 ) continue;

      /// Barrel
      if ( stackDetId.isBarrel() )
      {
        /// Get the Stack, iPhi, iZ from StackedTrackerDetId
        uint32_t iStack = stackDetId.iLayer();
        uint32_t iPhi   = stackDetId.iPhi();
        uint32_t iZ     = stackDetId.iZ();
        mapLayer_hPixelDigiRate_All[ iStack ]->Fill( iZ, iPhi );
      }
      /// Endcap
      else if ( stackDetId.isEndcap() )
      {
        /// Get the Stack, iPhi, iZ from StackedTrackerDetId
        uint32_t iSide  = stackDetId.iSide();
        uint32_t iStack = stackDetId.iDisk();
        uint32_t iRing  = stackDetId.iRing();
        uint32_t iPhi   = stackDetId.iPhi();
  
        std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
        mapDisk_hPixelDigiRate_All[ mapKey ]->Fill( iRing, iPhi );
      }
    }
  }

  /////////////////////////
  /// GET TRACK TRIGGER ///
  /////////////////////////
  edm::Handle< L1TkCluster_PixelDigi_Collection > PixelDigiL1TkClusterHandle;
  edm::Handle< L1TkStub_PixelDigi_Collection >    PixelDigiL1TkStubHandle;
  edm::Handle< L1TkStub_PixelDigi_Collection >    PixelDigiL1TkFailedStubHandle;
  iEvent.getByLabel( "L1TkClustersFromPixelDigis",             PixelDigiL1TkClusterHandle );
  iEvent.getByLabel( "L1TkStubsFromPixelDigis", "StubsPass",   PixelDigiL1TkStubHandle );
  iEvent.getByLabel( "L1TkStubsFromPixelDigis", "StubsFail",   PixelDigiL1TkFailedStubHandle );

  /// Loop over L1TkClusters
  L1TkCluster_PixelDigi_Collection::const_iterator iterL1TkCluster;
  for ( iterL1TkCluster = PixelDigiL1TkClusterHandle->begin();
        iterL1TkCluster != PixelDigiL1TkClusterHandle->end();
        ++iterL1TkCluster )
  {
    StackedTrackerDetId detIdClu( iterL1TkCluster->getDetId() );
    //unsigned int memberClu = iterL1TkCluster->getStackMember();
    bool genuineClu     = iterL1TkCluster->isGenuine();
    bool combinClu      = iterL1TkCluster->isCombinatoric();
    bool unknownClu     = iterL1TkCluster->isUnknown();

    /// Barrel
    if ( detIdClu.isBarrel() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = detIdClu.iLayer();
      uint32_t iPhi   = detIdClu.iPhi();
      uint32_t iZ     = detIdClu.iZ();
      mapLayer_hL1TkClusterRate_All[ iStack ]->Fill( iZ, iPhi );
      if ( genuineClu )
      {
        mapLayer_hL1TkClusterRate_Good[ iStack ]->Fill( iZ, iPhi );      
      }
      else if ( combinClu )
      {
        mapLayer_hL1TkClusterRate_Comb[ iStack ]->Fill( iZ, iPhi );      
      }
      else if ( unknownClu )
      {
        mapLayer_hL1TkClusterRate_Unkn[ iStack ]->Fill( iZ, iPhi );      
      }
    }
    /// Endcap
    else if ( detIdClu.isEndcap() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = detIdClu.iSide();
      uint32_t iStack = detIdClu.iDisk();
      uint32_t iRing  = detIdClu.iRing();
      uint32_t iPhi   = detIdClu.iPhi();
    
      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
      mapDisk_hL1TkClusterRate_All[ mapKey ]->Fill( iRing, iPhi );
      if ( genuineClu )
      {
        mapDisk_hL1TkClusterRate_Good[ mapKey ]->Fill( iRing, iPhi );      
      }
      else if ( combinClu )
      {
        mapDisk_hL1TkClusterRate_Comb[ mapKey ]->Fill( iRing, iPhi );      
      }
      else if ( unknownClu )
      {
        mapDisk_hL1TkClusterRate_Unkn[ mapKey ]->Fill( iRing, iPhi );      
      }
    }
  }

  /// Loop over L1TkStubs
  L1TkStub_PixelDigi_Collection::const_iterator iterL1TkStub;
  for ( iterL1TkStub = PixelDigiL1TkStubHandle->begin();
        iterL1TkStub != PixelDigiL1TkStubHandle->end();
        ++iterL1TkStub )
  {
    StackedTrackerDetId detIdStub( iterL1TkStub->getDetId() );
    bool genuineStub    = iterL1TkStub->isGenuine();
    bool combinStub     = iterL1TkStub->isCombinatoric();
    bool unknownStub    = iterL1TkStub->isUnknown();


    /// Barrel
    if ( detIdStub.isBarrel() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = detIdStub.iLayer();
      uint32_t iPhi   = detIdStub.iPhi();
      uint32_t iZ     = detIdStub.iZ();

      mapLayer_hL1TkStubRate_All[ iStack ]->Fill( iZ, iPhi );
      if ( genuineStub )
      {
        mapLayer_hL1TkStubRate_Good[ iStack ]->Fill( iZ, iPhi );      
      }
      else if ( combinStub )
      {
        mapLayer_hL1TkStubRate_Comb[ iStack ]->Fill( iZ, iPhi );      
      }
      else if ( unknownStub )
      {
        mapLayer_hL1TkStubRate_Unkn[ iStack ]->Fill( iZ, iPhi );      
      }
    }
    /// Endcap
    else if ( detIdStub.isEndcap() )
    {
      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = detIdStub.iSide();
      uint32_t iStack = detIdStub.iDisk();
      uint32_t iRing  = detIdStub.iRing();
      uint32_t iPhi   = detIdStub.iPhi();

      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iSide, iStack );
      mapDisk_hL1TkStubRate_All[ mapKey ]->Fill( iRing, iPhi );
      if ( genuineStub )
      {
        mapDisk_hL1TkStubRate_Good[ mapKey ]->Fill( iRing, iPhi );      
      }
      else if ( combinStub )
      {
        mapDisk_hL1TkStubRate_Comb[ mapKey ]->Fill( iRing, iPhi );      
      }
      else if ( unknownStub )
      {
        mapDisk_hL1TkStubRate_Unkn[ mapKey ]->Fill( iRing, iPhi );      
      }
    }
  }


} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(ValidateRates);

