#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiTester.h"


EcalDigiTester::EcalDigiTester (const edm::ParameterSet& params) 
{
  theParameterMap = new EcalSimParameterMap () ;
  std::cout << "[EcalDigiTester][ctor] etering\n" ;
}


EcalDigiTester::~EcalDigiTester () 
{
  delete theParameterMap ;
}


void EcalDigiTester::analyze (const edm::Event& event, const edm::EventSetup& eventSetup) 
{
//  edm::ESHandle<EcalPedestals> pedHandle ;
//  eventSetup.get<EcalPedestalsRcd> ().get ( pedHandle ) ;
//  theCoder->setPedestals (pedHandle.product ()) ;

  std::cout << "[EcalDigiTester][analyze] ----------------------------\n" ;

  checkCalibrations (eventSetup) ;
  checkGeometry (eventSetup) ;

  // Get the hits
  // ------------
  
  edm::Handle<CrossingFrame> crossingFrame ;
  event.getByType (crossingFrame) ;

  // access to SimHits
  const std::string barrelHitsName ("EcalHitsEB") ;
  const std::string endcapHitsName ("EcalHitsEE") ;

  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits (new MixCollection<PCaloHit>(crossingFrame.product (), barrelHitsName)) ;
//  std::auto_ptr<MixCollection<PCaloHit> > 
//    endcapHits (new MixCollection<PCaloHit> (crossingFrame.product (), endcapHitsName)) ;

  std::vector<simpleUnit> EBcheckhits ;
  
  // loop over the EB hits
  for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin () ;
       hitItr != barrelHits->end () ;
       ++hitItr)
  {
    EBDetId detId (hitItr->id ()) ;
    std::cout << "[EcalDigiTester][analyze] hit"
              << "\tE: " << hitItr->energy () 
              << "\teta: " << detId.ieta () 
              << "\tphi: " << detId.iphi () 
              << "\n" ;
    const CaloCellGeometry* cellGeometry =
      theGeometry->getSubdetectorGeometry (detId)->getGeometry (detId) ;

    double eta = cellGeometry->getPosition ().eta () ;
    double phi = cellGeometry->getPosition ().phi () ;
    double E = hitItr->energy () ;
    double cosTheta = cos (cellGeometry->getPosition ().theta ()) ;
    double ET = hitItr->energy () * cosTheta ;
    
    EBcheckhits.push_back (simpleUnit (eta,phi,E)) ;
  } // loop over the EB hits

  std::sort (EBcheckhits.begin (), EBcheckhits.end (), compare) ;

  // Get the digis
  // -------------
     
  edm::Handle<EBDigiCollection> barrelResultHandle ;
  event.getByType (barrelResultHandle) ;
 
  const EBDigiCollection * barrelResult = barrelResultHandle.product () ;
  std::vector<simpleUnit> EBcheckdigis ;

  // loop over the digis
  for (std::vector<EBDataFrame>::const_iterator digis = barrelResult->begin () ;
       digis != barrelResult->end () ;
       ++digis)
    {
      
      EBDetId detId = digis->id () ;
 
      const CaloCellGeometry* cellGeometry =
        theGeometry->getSubdetectorGeometry (detId)->getGeometry (detId) ;
 
      double eta = cellGeometry->getPosition ().eta () ;
      double phi = cellGeometry->getPosition ().phi () ;
      double cosTheta = cos (cellGeometry->getPosition ().theta ()) ;
 
      double Emax = -1 ;
      for (int sample = 0 ; sample < digis->size () ; ++sample)
        {
          double value = digis->sample (sample).adc () ;
          int gainId = digis->sample (sample).gainId () ;
          if (Emax < value) Emax = value ;
        }
 
      EBcheckdigis.push_back (simpleUnit (eta,phi,Emax)) ;
    
    } // loop over the digis

  std::sort (EBcheckdigis.begin (), EBcheckdigis.end (), compare) ;

  // print results
  // -------------

  edm::LogInfo ("EcalDigiTester") << "EB Digis: " << barrelResult->size () ;

  CaloDigiCollectionSorter sorter (5) ;
  std::vector<EBDataFrame> sortedDigis = sorter.sortedVector (*barrelResult) ;
  std::cout << "Top 10 EB digis" << std::endl ;
  for (int i = 0 ; i < std::min (10, (int) sortedDigis.size ()) ; ++i) 
   {
//    double ET = hitItr->energy () * cosTheta ;
//     std::cout << "[EcalDigiTester][analyze] digi " << i
//               << "\t" << sortedDigis[i] ;
   }

}


void  EcalDigiTester::checkCalibrations (const edm::EventSetup & eventSetup) 
{}


void EcalDigiTester::checkGeometry (const edm::EventSetup & eventSetup) 
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry ;
  eventSetup.get<IdealGeometryRecord> ().get (geometry) ;

  theGeometry = &*geometry ;
  
///  const CaloGeometry * pGeometry = &*hGeometry;
///  void setGeometry(const CaloGeometry * geometry) { theGeometry = geometry; }

  theBarrelDets.clear () ;
  theEndcapDets.clear () ;

  theBarrelDets =  geometry->getValidDetIds (DetId::Ecal, EcalBarrel) ;
  theEndcapDets =  geometry->getValidDetIds (DetId::Ecal, EcalEndcap) ;

  //PG FIXME
  std::cout << "deb geometry: "
            << "\t barrel: " << theBarrelDets.size () 
            << "\t endcap: " << theEndcapDets.size () 
            << std::endl ;
            
}


// ------------------------------------------------------------------


simpleUnit::simpleUnit (double eta,double phi,double E) : 
    m_eta (eta) ,
    m_phi (phi) ,
    m_E (E) {} ;
simpleUnit::~simpleUnit () {} ;
bool simpleUnit::operator< (const simpleUnit& altro) const 
  {return m_E < altro.m_E ;}

std::ostream & operator<< (std::ostream & os, const simpleUnit & su) 
{
  os << "[SU]\t" << su.m_eta << "\t" << su.m_phi << "\t" << su.m_E << std::endl ;
  return os;
}



// ------------------------------------------------------------------


bool compare (const simpleUnit& primo, const simpleUnit secondo) 
 {return primo.m_E > secondo.m_E ; }

