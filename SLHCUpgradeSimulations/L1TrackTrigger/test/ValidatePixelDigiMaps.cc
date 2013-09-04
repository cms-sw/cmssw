////////////////////////////
// Geometry Checklist     //
// Maps with PixelDigis   //
//                        //
// Nicola Pozzobon - 2012 //
////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH2D.h>

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class ValidatePixelDigiMaps : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit ValidatePixelDigiMaps(const edm::ParameterSet& iConfig);
    virtual ~ValidatePixelDigiMaps();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:
    /// Global Position of SimHits
    TH2D* hPixelDigi_Barrel_XY;
    TH2D* hPixelDigi_Barrel_XY_Zoom;
    TH2D* hPixelDigi_Endcap_Fw_XY;
    TH2D* hPixelDigi_Endcap_Bw_XY;
    TH2D* hPixelDigi_RZ;
    TH2D* hPixelDigi_Endcap_Fw_RZ_Zoom;
    TH2D* hPixelDigi_Endcap_Bw_RZ_Zoom;

    std::map< std::string, TH2D* > m_hPixelDigi_Barrel_XY_Survey;
    std::map< std::string, TH2D* > m_hPixelDigi_RZ_Survey;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
ValidatePixelDigiMaps::ValidatePixelDigiMaps(edm::ParameterSet const& iConfig) 
{
  /// Insert here what you need to initialize
}

/////////////
// DESTRUCTOR
ValidatePixelDigiMaps::~ValidatePixelDigiMaps()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void ValidatePixelDigiMaps::endJob()
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " ValidatePixelDigiMaps::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void ValidatePixelDigiMaps::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " ValidatePixelDigiMaps::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service<TFileService> fs;

  hPixelDigi_Barrel_XY          = fs->make<TH2D>( "hPixelDigi_Barrel_XY",         "PixelDigi Barrel y vs. x",              960, -120, 120, 960, -120, 120 );
  hPixelDigi_Barrel_XY_Zoom     = fs->make<TH2D>( "hPixelDigi_Barrel_XY_Zoom",    "PixelDigi Barrel y vs. x",              960, 30, 60, 960, -15, 15 );
  hPixelDigi_Endcap_Fw_XY       = fs->make<TH2D>( "hPixelDigi_Endcap_Fw_XY",      "PixelDigi Forward Endcap y vs. x",      960, -120, 120, 960, -120, 120 );
  hPixelDigi_Endcap_Bw_XY       = fs->make<TH2D>( "hPixelDigi_Endcap_Bw_XY",      "PixelDigi Backward Endcap y vs. x",     960, -120, 120, 960, -120, 120 );
  hPixelDigi_RZ                 = fs->make<TH2D>( "hPixelDigi_RZ",                "PixelDigi #rho vs. z",                  900, -300, 300, 480, 0, 120 );
  hPixelDigi_Endcap_Fw_RZ_Zoom  = fs->make<TH2D>( "hPixelDigi_Endcap_Fw_RZ_Zoom", "PixelDigi Forward Endcap #rho vs. z",   960, 140, 170, 960, 30, 60 );
  hPixelDigi_Endcap_Bw_RZ_Zoom  = fs->make<TH2D>( "hPixelDigi_Endcap_Bw_RZ_Zoom", "PixelDigi Backward Endcap #rho vs. z",  960, -170, -140, 960, 70, 100 );

  hPixelDigi_Barrel_XY->Sumw2();
  hPixelDigi_Barrel_XY_Zoom->Sumw2();
  hPixelDigi_Endcap_Fw_XY->Sumw2();
  hPixelDigi_Endcap_Bw_XY->Sumw2();
  hPixelDigi_RZ->Sumw2();
  hPixelDigi_Endcap_Fw_RZ_Zoom->Sumw2();
  hPixelDigi_Endcap_Bw_RZ_Zoom->Sumw2();

  for ( int ix = 0; ix < 11; ix ++ )
  {
    for ( int iy = 0; iy < 11; iy++ )
    {
      histoName.str("");
      histoTitle.str("");
      histoName << "hPixelDigi_Barrel_XY_Survey_" << -110+ix*20 << "x" << -110+(1+ix)*20 << "_" << -110+iy*20 << "y" << -110+(1+iy)*20;
      histoTitle << "PixelDigi Barrel y (" << -110+iy*20 << ", " << -110+(1+iy)*20 << ") cm vs. x (" << -110+ix*20 << ", " << -110+(1+ix)*20 <<") cm";
      TH2D* h = fs->make<TH2D>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -110+ix*20, -110+(1+ix)*20, 800, -110+iy*20, -110+(1+iy)*20 );
      m_hPixelDigi_Barrel_XY_Survey.insert( std::pair< std::string, TH2D* >( histoName.str(), h ) );
    }
  }

  for ( int ir = 0; ir < 6; ir ++ )
  {
    for ( int iz = 0; iz < 27; iz++ )
    {
      histoName.str("");
      histoTitle.str("");
      histoName << "hPixelDigi_RZ_Survey_" << -10+ir*20 << "r" << -10+(1+ir)*20 << "_" << -270+iz*20 << "z" << -270+(1+iz)*20;
      histoTitle << "PixelDigi #rho (" << -10+ir*20 << ", " << -10+(1+ir)*20 << ") cm vs. z (" << -270+iz*20 << ", " << -270+(1+iz)*20 <<") cm";
      TH2D* h = fs->make<TH2D>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -270+iz*20, -270+(1+iz)*20, 800, -10+ir*20, -10+(1+ir)*20 );
      m_hPixelDigi_RZ_Survey.insert( std::pair< std::string, TH2D* >( histoName.str(), h ) );
    }
  }

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void ValidatePixelDigiMaps::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Geometry handles etc
  edm::ESHandle< TrackerGeometry >         geometryHandle;
  const TrackerGeometry*                   theGeometry;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  theGeometry = &(*geometryHandle);

  /////////////////////
  // GET PIXEL DIGIS //
  /////////////////////
  edm::Handle< edm::DetSetVector< PixelDigi > >         PixelDigiHandle;
  edm::Handle< edm::DetSetVector< PixelDigiSimLink > >  PixelDigiSimLinkHandle;
  iEvent.getByLabel( "simSiPixelDigis", PixelDigiHandle );
  iEvent.getByLabel( "simSiPixelDigis", PixelDigiSimLinkHandle );

  edm::DetSetVector<PixelDigi>::const_iterator detsIter;
  edm::DetSet<PixelDigi>::const_iterator       hitsIter;

  /// Loop over detector elements identifying PixelDigis
  for ( detsIter = PixelDigiHandle->begin();
        detsIter != PixelDigiHandle->end();
        detsIter++ )
  {
    DetId tkId = detsIter->id;

    /// Loop over Digis in this specific detector element
    for ( hitsIter = detsIter->data.begin();
          hitsIter != detsIter->data.end();
          hitsIter++ )
    {
      /// Threshold (here it is NOT redundant)
      if ( hitsIter->adc() <= 30 ) continue;
      /// Try to learn something from PixelDigi position
      const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( tkId );
      MeasurementPoint mp( hitsIter->row() + 0.5, hitsIter->column() + 0.5 );
      GlobalPoint pdPos = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;

      std::ostringstream histoNameXY;
      std::ostringstream histoNameRZ;
      histoNameXY << "hPixelDigi_Barrel_XY_Survey_" << (20*(floor((pdPos.x()+10)/20))-10) << "x" << (20*(floor((pdPos.x()+10)/20))+10) <<
                                                "_" << (20*(floor((pdPos.y()+10)/20))-10) << "y" << (20*(floor((pdPos.y()+10)/20))+10);
      histoNameRZ << "hPixelDigi_RZ_Survey_" << (20*(floor((pdPos.perp()+10)/20))-10) << "r" << (20*(floor((pdPos.perp()+10)/20))+10) <<
                                         "_" << (20*(floor((pdPos.z()+10)/20))-10) << "z" << (20*(floor((pdPos.z()+10)/20))+10);
      hPixelDigi_RZ->Fill( pdPos.z(), pdPos.perp() );
      if( m_hPixelDigi_RZ_Survey.find( histoNameRZ.str() ) != m_hPixelDigi_RZ_Survey.end() )
        m_hPixelDigi_RZ_Survey.find( histoNameRZ.str() )->second->Fill( pdPos.z(), pdPos.perp() );
      if (gDetUnit->type().isBarrel())
      {
        hPixelDigi_Barrel_XY->Fill( pdPos.x(), pdPos.y() );
        hPixelDigi_Barrel_XY_Zoom->Fill( pdPos.x(), pdPos.y() );
        if ( m_hPixelDigi_Barrel_XY_Survey.find( histoNameXY.str() ) != m_hPixelDigi_Barrel_XY_Survey.end() )
          m_hPixelDigi_Barrel_XY_Survey.find( histoNameXY.str() )->second->Fill( pdPos.x(), pdPos.y() );
      }
      else if (gDetUnit->type().isEndcap())
      {
        if (pdPos.z() > 0)
        {
          hPixelDigi_Endcap_Fw_XY->Fill( pdPos.x(), pdPos.y() );
          hPixelDigi_Endcap_Fw_RZ_Zoom->Fill( pdPos.z(), pdPos.perp() );
        }
        else
        {
          hPixelDigi_Endcap_Bw_XY->Fill( pdPos.x(), pdPos.y() );
          hPixelDigi_Endcap_Bw_RZ_Zoom->Fill( pdPos.z(), pdPos.perp() );
        }
      }
    }
  }

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(ValidatePixelDigiMaps);

