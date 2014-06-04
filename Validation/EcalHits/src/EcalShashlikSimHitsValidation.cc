/*
 * \file EcalShashlikSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
*/

#include <DataFormats/EcalDetId/interface/EKDetId.h>
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/EcalHits/interface/EcalShashlikSimHitsValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"


using namespace cms;
using namespace edm;
using namespace std;

EcalShashlikSimHitsValidation::EcalShashlikSimHitsValidation(const edm::ParameterSet& ps):
  g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
  EKHitsCollection(ps.getParameter<std::string>("EKHitsCollection")),
  ValidationCollection(ps.getParameter<std::string>("ValidationCollection")),
  _topology(NULL)
{   

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
 
  // get hold of back-end interface
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();           
  if ( dbe_ ) {
    if ( verbose_ ) { dbe_->setVerbose(1); } 
    else            { dbe_->setVerbose(0); }
  }
                                                                                                            
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }
 

  meEKzpHits_             = 0;
  meEKzmHits_             = 0;
  meEKzpCrystals_         = 0;
  meEKzmCrystals_         = 0;
  meEKzpOccupancy_        = 0;
  meEKzmOccupancy_        = 0;
  meEKLongitudinalShower_ = 0;
  meEKHitEnergy_          = 0;
  meEKhitLog10Energy_       = 0;
  meEKhitLog10EnergyNorm_   = 0;
  meEKhitLog10Energy25Norm_ = 0;
  meEKHitEnergy2_         = 0;
  meEKcrystalEnergy_      = 0;
  meEKcrystalEnergy2_     = 0;

  meEKe1_  = 0;  
  meEKe4_  = 0;  
  meEKe9_  = 0;  
  meEKe16_ = 0; 
  meEKe25_ = 0; 

  meEKe1oe4_   = 0;  
  meEKe1oe9_   = 0;  
  meEKe4oe9_   = 0;  
  meEKe9oe16_  = 0;
  meEKe1oe25_  = 0;  
  meEKe9oe25_  = 0; 
  meEKe16oe25_ = 0;

  myEntries = 0;
  for ( int myStep = 0; myStep<26; myStep++) { eRLength[myStep] = 0.0; }

  Char_t histo[200];
 
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalHitsV/EcalSimHitsValidation");
  
    sprintf (histo, "EK+ hits multiplicity" ) ;
    meEKzpHits_ = dbe_->book1D(histo, histo, 50, 0., 5000.) ; 

    sprintf (histo, "EK- hits multiplicity" ) ;
    meEKzmHits_ = dbe_->book1D(histo, histo, 50, 0., 5000.) ; 

    sprintf (histo, "EK+ crystals multiplicity" ) ;
    meEKzpCrystals_ = dbe_->book1D(histo, histo, 200, 0., 2000.) ; 

    sprintf (histo, "EK- crystals multiplicity" ) ;
    meEKzmCrystals_ = dbe_->book1D(histo, histo, 200, 0., 2000.) ; 

    sprintf (histo, "EK+ occupancy" ) ;
    meEKzpOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
    sprintf (histo, "EK- occupancy" ) ;
    meEKzmOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
    sprintf (histo, "EK longitudinal shower profile" ) ;
    meEKLongitudinalShower_ = dbe_->bookProfile(histo, histo, 26,0,26, 100, 0, 20000);

    sprintf (histo, "EK hits energy spectrum" );
    meEKHitEnergy_ = dbe_->book1D(histo, histo, 4000, 0., 400.);

    sprintf (histo, "EK hits log10energy spectrum" );
    meEKhitLog10Energy_ = dbe_->book1D(histo, histo, 140, -10., 4.);

    sprintf (histo, "EK hits log10energy spectrum vs normalized energy" );
    meEKhitLog10EnergyNorm_ = dbe_->bookProfile(histo, histo, 140, -10., 4., 100, 0., 1.);

    sprintf (histo, "EK hits log10energy spectrum vs normalized energy25" );
    meEKhitLog10Energy25Norm_ = dbe_->bookProfile(histo, histo, 140, -10., 4., 100, 0., 1.);

    sprintf (histo, "EK hits energy spectrum 2" );
    meEKHitEnergy2_ = dbe_->book1D(histo, histo, 1000, 0., 0.001);

    sprintf (histo, "EK crystal energy spectrum" );
    meEKcrystalEnergy_ = dbe_->book1D(histo, histo, 5000, 0., 50.);

    sprintf (histo, "EK crystal energy spectrum 2" );
    meEKcrystalEnergy2_ = dbe_->book1D(histo, histo, 1000, 0., 0.001);

    sprintf (histo, "EK E1" ) ;
    meEKe1_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EK E4" ) ;
    meEKe4_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EK E9" ) ;
    meEKe9_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EK E16" ) ;
    meEKe16_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EK E25" ) ;
    meEKe25_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EK E1oE4" ) ;
    meEKe1oe4_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EK E1oE9" ) ;
    meEKe1oe9_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EK E4oE9" ) ;
    meEKe4oe9_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EK E9oE16" ) ;
    meEKe9oe16_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EK E1oE25" ) ;
    meEKe1oe25_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EK E9oE25" ) ;
    meEKe9oe25_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EK E16oE25" ) ;
    meEKe16oe25_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);
  }
}

EcalShashlikSimHitsValidation::~EcalShashlikSimHitsValidation(){

}

void EcalShashlikSimHitsValidation::beginJob(){

}

void EcalShashlikSimHitsValidation::endJob(){

  //for ( int myStep = 0; myStep<26; myStep++){
  //  if (meEKLongitudinalShower_) meEKLongitudinalShower_->Fill(float(myStep), eRLength[myStep]/myEntries);
  //}

}

void EcalShashlikSimHitsValidation::analyze(const edm::Event& e, const edm::EventSetup& c){

  //edm::ESHandle<ShashlikTopology> topo;
  //c.get<ShashlikNumberingRecord>().get(topo);
  //  _topology = &*topo;

  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
  edm::Handle<edm::PCaloHitContainer> EcalHitsEK;
  e.getByLabel(g4InfoLabel,EKHitsCollection,EcalHitsEK);

  // Do nothing if no Shashlik data available
  if( ! EcalHitsEK.isValid() ) return;

  edm::Handle<PEcalValidInfo> MyPEcalValidInfo;
  e.getByLabel(g4InfoLabel,ValidationCollection,MyPEcalValidInfo);

  std::vector<PCaloHit> theEKCaloHits;
  theEKCaloHits.insert(theEKCaloHits.end(), EcalHitsEK->begin(), EcalHitsEK->end());
  
  myEntries++;

  std::map<unsigned int, std::vector<PCaloHit*>,std::less<unsigned int> > CaloHitMap;
  
  double EKetzp_ = 0.;
  double EKetzm_ = 0.;
  
  double ee1  = 0.0;
  double ee4  = 0.0;
  double ee9  = 0.0;
  double ee16 = 0.0;
  double ee25 = 0.0;
  std::vector<double> econtr(140, 0. );
  std::vector<double> econtr25(140, 0. );
  
  MapType eemap;
  MapType eemapzp;
  MapType eemapzm;
  uint32_t nEKzpHits = 0;
  uint32_t nEKzmHits = 0;
  
  for (std::vector<PCaloHit>::iterator isim = theEKCaloHits.begin();
       isim != theEKCaloHits.end(); ++isim){

    if ( isim->time() > 500. ) { continue; }

    CaloHitMap[ isim->id()].push_back(&(*isim));
    
    EKDetId eeid (isim->id()) ;
    
    LogDebug("HitInfo") 
      << " CaloHit " << isim->getName() << "\n" 
      << " DetID = "<<isim->id()<< " EKDetId = " << eeid.ix() << " " << eeid.iy() << "\n"	
      << " Time = " << isim->time() << "\n"
      << " Track Id = " << isim->geantTrackId() << "\n"
      << " Energy = " << isim->energy();
    
    uint32_t crystid = eeid.rawId();
     
    if (eeid.zside() > 0 ) {
      nEKzpHits++;
      EKetzp_ += isim->energy();
      eemapzp[crystid] += isim->energy();
      if (meEKzpOccupancy_) meEKzpOccupancy_->Fill(eeid.ix(), eeid.iy());
    }
    else if (eeid.zside() < 0 ) {
      nEKzmHits++;
      EKetzm_ += isim->energy();
      eemapzm[crystid] += isim->energy();
      if (meEKzmOccupancy_) meEKzmOccupancy_->Fill(eeid.ix(), eeid.iy());
    }

    if (meEKHitEnergy_) meEKHitEnergy_->Fill(isim->energy());
    if( isim->energy() > 0 ) {
      if( meEKhitLog10Energy_ ) meEKhitLog10Energy_->Fill(log10(isim->energy()));
      int log10i = int( ( log10(isim->energy()) + 10. ) * 10. );
      if( log10i >=0 && log10i < 140 ) econtr[log10i] += isim->energy();
    }
    if (meEKHitEnergy2_) meEKHitEnergy2_->Fill(isim->energy());
    eemap[crystid] += isim->energy();


  }
  
  if (meEKzpCrystals_) meEKzpCrystals_->Fill(eemapzp.size());
  if (meEKzmCrystals_) meEKzmCrystals_->Fill(eemapzm.size());
  
  if (meEKcrystalEnergy_) {
    for (std::map<uint32_t,float,std::less<uint32_t> >::iterator it = eemap.begin(); it != eemap.end(); ++it ) meEKcrystalEnergy_->Fill((*it).second);
  }
  if (meEKcrystalEnergy2_) {
    for (std::map<uint32_t,float,std::less<uint32_t> >::iterator it = eemap.begin(); it != eemap.end(); ++it ) meEKcrystalEnergy2_->Fill((*it).second);
  }
    
  if (meEKzpHits_)    meEKzpHits_->Fill(nEKzpHits);
  if (meEKzmHits_)    meEKzmHits_->Fill(nEKzmHits);
    
  
  int nEKHits = nEKzmHits + nEKzpHits;
  if (nEKHits > 0) {
    
    uint32_t  eecenterid = getUnitWithMaxEnergy(eemap);
    EKDetId myEKid(eecenterid);
    int bx = myEKid.ix();
    int by = myEKid.iy();
    int bz = myEKid.zside();
    ee1 =  energyInMatrixEK(1,1,bx,by,bz,eemap);
    if (meEKe1_) meEKe1_->Fill(ee1);
    ee9 =  energyInMatrixEK(3,3,bx,by,bz,eemap);
    if (meEKe9_) meEKe9_->Fill(ee9);
    ee25=  energyInMatrixEK(5,5,bx,by,bz,eemap);
    if (meEKe25_) meEKe25_->Fill(ee25);
    
    std::vector<uint32_t> ids25; ids25 = getIdsAroundMax(5,5,bx,by,bz,eemap);

    for( unsigned i=0; i<25; i++ ) {
      for( unsigned int j=0; j<CaloHitMap[ids25[i]].size(); j++ ) {
	if( CaloHitMap[ids25[i]][j]->energy() > 0 ) {
	  int log10i = int( ( log10( CaloHitMap[ids25[i]][j]->energy()) + 10. ) * 10. );
	  if( log10i >=0 && log10i < 140 ) econtr25[log10i] += CaloHitMap[ids25[i]][j]->energy();
	}
      }
    }

    MapType  neweemap;
    if( fillEKMatrix(3,3,bx,by,bz,neweemap, eemap)){
      ee4 = eCluster2x2(neweemap);
      if (meEKe4_) meEKe4_->Fill(ee4);
    }
    if( fillEKMatrix(5,5,bx,by,bz,neweemap, eemap)){
      ee16 = eCluster4x4(ee9,neweemap); 
      if (meEKe16_) meEKe16_->Fill(ee16);
    }
    
    if (meEKe1oe4_   && ee4  > 0.1 ) meEKe1oe4_  ->Fill(ee1/ee4);
    if (meEKe1oe9_   && ee9  > 0.1 ) meEKe1oe9_  ->Fill(ee1/ee9);
    if (meEKe4oe9_   && ee9  > 0.1 ) meEKe4oe9_  ->Fill(ee4/ee9);
    if (meEKe9oe16_  && ee16 > 0.1 ) meEKe9oe16_ ->Fill(ee9/ee16);
    if (meEKe16oe25_ && ee25 > 0.1 ) meEKe16oe25_->Fill(ee16/ee25);
    if (meEKe1oe25_  && ee25 > 0.1 ) meEKe1oe25_ ->Fill(ee1/ee25);
    if (meEKe9oe25_  && ee25 > 0.1 ) meEKe9oe25_ ->Fill(ee9/ee25);

    if( meEKhitLog10EnergyNorm_ && (EKetzp_+EKetzm_) != 0 ) {
      for( int i=0; i<140; i++ ) {
	meEKhitLog10EnergyNorm_->Fill( -10.+(float(i)+0.5)/10., econtr[i]/(EKetzp_+EKetzm_) );
      }
    }

    if( meEKhitLog10Energy25Norm_ && ee25 != 0 ) {
      for( int i=0; i<140; i++ ) {
	meEKhitLog10Energy25Norm_->Fill( -10.+(float(i)+0.5)/10., econtr25[i]/ee25 );
      }
    }
    
  }
    
  if( MyPEcalValidInfo.isValid() ) {
    if ( MyPEcalValidInfo->ee1x1() > 0. ) {
      std::vector<float>  EX0 = MyPEcalValidInfo->eX0();
      if (meEKLongitudinalShower_) meEKLongitudinalShower_->Reset();
      for (int myStep=0; myStep< 26; myStep++ ) { 
	eRLength[myStep] += EX0[myStep]; 
	if (meEKLongitudinalShower_) meEKLongitudinalShower_->Fill(float(myStep), eRLength[myStep]/myEntries);
      }
    }
  }
}

float EcalShashlikSimHitsValidation::energyInMatrixEK(int nCellInX, int nCellInY,
                                        int centralX, int centralY,
                                        int centralZ, MapType& themap){
  
  int   ncristals   = 0;
  float totalEnergy = 0.;
        
  int goBackInX = nCellInX/2;
  int goBackInY = nCellInY/2;
  int startX    = centralX-goBackInX;
  int startY    = centralY-goBackInY;
  
  for (int ix=startX; ix<startX+nCellInX; ix++) {
    for (int iy=startY; iy<startY+nCellInY; iy++) {
      uint32_t index ;
      
//       if(_topology->validXY(ix,iy)) {
      index = EKDetId(ix,iy,0, 0,centralZ).rawId();
//       } else { continue; }

      totalEnergy   += themap[index];
      ncristals     += 1;
    }
  }
  
  LogDebug("GeomInfo")
    << nCellInX << " x " << nCellInY 
    << " EK matrix energy = " << totalEnergy
    << " for " << ncristals << " crystals";
  return totalEnergy;
  
}

std::vector<uint32_t> EcalShashlikSimHitsValidation::getIdsAroundMax( int nCellInX, int nCellInY, int centralX, int centralY, int centralZ, MapType& themap){
  
  int   ncristals   = 0;
  std::vector<uint32_t> ids( nCellInX*nCellInY );
        
  int goBackInX = nCellInX/2;
  int goBackInY = nCellInY/2;
  int startX    = centralX-goBackInX;
  int startY    = centralY-goBackInY;
  
  for (int ix=startX; ix<startX+nCellInX; ix++) {
    for (int iy=startY; iy<startY+nCellInY; iy++) {
      uint32_t index ;
      
//       if(_topology->validXY(ix,iy)) {
      index = EKDetId(ix,iy,0,0, centralZ).rawId();
//       }
//       else { continue; }

      ids[ncristals] = index;
      ncristals     += 1;
    }
  }
  
  return ids;
}

bool  EcalShashlikSimHitsValidation::fillEKMatrix(int nCellInX, int nCellInY,
                                    int CentralX, int CentralY,int CentralZ,
                                    MapType& fillmap, MapType&  themap)
{
   int goBackInX = nCellInX/2;
   int goBackInY = nCellInY/2;

   int startX  =  CentralX - goBackInX;
   int startY  =  CentralY - goBackInY;

   int i = 0 ;
   for ( int ix = startX; ix < startX+nCellInX; ix ++ ) {

      for( int iy = startY; iy < startY + nCellInY; iy++ ) {

        uint32_t index ;

// 	if(_topology->validXY(ix, iy)) {
	index = EKDetId(ix,iy,0, 0, CentralZ).rawId();
// 	}
// 	else { continue; }
        fillmap[i++] = themap[index];
      }
   }
   uint32_t  centerid = getUnitWithMaxEnergy(themap);

   if ( fillmap[i/2] == themap[centerid] ) 
        return true;
   else
        return false;
}


float EcalShashlikSimHitsValidation::eCluster2x2( MapType& themap){
  float  E22=0.;
  float e012  = themap[0]+themap[1]+themap[2];
  float e036  = themap[0]+themap[3]+themap[6];
  float e678  = themap[6]+themap[7]+themap[8];
  float e258  = themap[2]+themap[5]+themap[8];

  if ( (e012>e678 || e012==e678) && (e036>e258  || e036==e258))
    return     E22=themap[0]+themap[1]+themap[3]+themap[4];
  else if ( (e012>e678 || e012==e678)  && (e036<e258 || e036==e258) )
    return    E22=themap[1]+themap[2]+themap[4]+themap[5];
  else if ( (e012<e678 || e012==e678) && (e036>e258 || e036==e258))
    return     E22=themap[3]+themap[4]+themap[6]+themap[7];
  else if ( (e012<e678|| e012==e678)  && (e036<e258|| e036==e258) )
    return    E22=themap[4]+themap[5]+themap[7]+themap[8];
  else {
    return E22;
  }
}

float EcalShashlikSimHitsValidation::eCluster4x4(float e33,  MapType&  themap){
  float E44=0.;
  float e0_4   = themap[0]+themap[1]+themap[2]+themap[3]+themap[4];
  float e0_20  = themap[0]+themap[5]+themap[10]+themap[15]+themap[20];
  float e4_24  = themap[4]+themap[9]+themap[14]+themap[19]+themap[24];
  float e0_24  = themap[20]+themap[21]+themap[22]+themap[23]+themap[24];
  
  if ((e0_4>e0_24 || e0_4==e0_24) && (e0_20>e4_24|| e0_20==e4_24))
    return E44=e33+themap[0]+themap[1]+themap[2]+themap[3]+themap[5]+themap[10]+themap[15];
  else if ((e0_4>e0_24 || e0_4==e0_24)  && (e0_20<e4_24 || e0_20==e4_24))
    return E44=e33+themap[1]+themap[2]+themap[3]+themap[4]+themap[9]+themap[14]+themap[19];
  else if ((e0_4<e0_24|| e0_4==e0_24) && (e0_20>e4_24 || e0_20==e4_24))
    return E44=e33+themap[5]+themap[10]+themap[15]+themap[20]+themap[21]+themap[22]+themap[23];
  else if ((e0_4<e0_24|| e0_4==e0_24) && (e0_20<e4_24 || e0_20==e4_24))
    return E44=e33+themap[21]+themap[22]+themap[23]+themap[24]+themap[9]+themap[14]+themap[19];
  else{
    return E44;
  }
}

uint32_t EcalShashlikSimHitsValidation::getUnitWithMaxEnergy(MapType& themap) {
  
  //look for max
  uint32_t unitWithMaxEnergy = 0;
  float    maxEnergy = 0.;
  
  MapType::iterator iter;
  for (iter = themap.begin(); iter != themap.end(); iter++) {
    
    if (maxEnergy < (*iter).second) {
      maxEnergy = (*iter).second;       
      unitWithMaxEnergy = (*iter).first;
    }                           
  }
  
  LogDebug("GeomInfo")
    << " max energy of " << maxEnergy 
    << " GeV in Unit id " << unitWithMaxEnergy;
  return unitWithMaxEnergy;
}


DEFINE_FWK_MODULE(EcalShashlikSimHitsValidation);
