#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "vdt/vdtMath.h"

using namespace hgc_digi;
using namespace hgc_digi_utils;


HGCHEbackSignalScaler::HGCHEbackSignalScaler(const CaloSubdetectorGeometry* geom, const std::string& fullpath)
{
  hgcalGeom_ = static_cast<const HGCalGeometry*>(geom);
  doseMap_ = readDosePars(fullpath);
}

std::map<int, HGCHEbackSignalScaler::DoseParameters> HGCHEbackSignalScaler::readDosePars(const std::string& fullpath)
{
  std::map<int, DoseParameters> result;

  //no dose file means no aging
  if(fullpath == "")
    return result;

  edm::FileInPath fp(fullpath);
  std::ifstream infile(fp.fullPath());
  if(!infile.is_open())
  {
    throw cms::Exception("FileNotFound") << "Unable to open '" << fullpath << "'" << std::endl;
  }
  std::string line;
  while(getline(infile,line))
  {
    int layer;
    DoseParameters dosePars;

    //space-separated
    std::stringstream linestream(line);
    linestream >> layer >> dosePars.a_ >>  dosePars.b_ >> dosePars.c_;

    result[layer] = dosePars;
  }
  return result;
}

double HGCHEbackSignalScaler::getDoseValue(const HGCScintillatorDetId& cellId)
{
  float radius = computeRadius(cellId) / 100.; //radius in m
  int layer = cellId.layer();
  double cellDose = std::pow(10, doseMap_[layer].a_ + doseMap_[layer].b_*radius + doseMap_[layer].c_*std::pow(radius, 2)); //dose in rad
  return cellDose/1000.; //convert to kRad
}

float HGCHEbackSignalScaler::scaleByDose(const HGCScintillatorDetId& cellId)
{
  if(doseMap_.empty())
    return 1.;

  double cellDose = getDoseValue(cellId); //in kRad
  double scaleFactor = std::exp( -std::pow(cellDose, 0.65) / 199.6 );

  if(verbose_)
  {
    int layer = cellId.layer();
    std::cout << "HGCHEbackSignalScaler::scaleByDose - layer, a, b, c: "
              << layer << " "
              << doseMap_[layer].a_ << " "
              << doseMap_[layer].b_ << " "
              << doseMap_[layer].c_ << std::endl;

    std::cout << "HGCHEbackSignalScaler::scaleByDose - Dose, scaleFactor: "
              << cellDose << " "
              << scaleFactor << std::endl;
  }

  return scaleFactor;
}


float HGCHEbackSignalScaler::scaleByArea(const HGCScintillatorDetId& cellId)
{
  float edge = computeEdge(cellId);
  float scaleFactor = 3. / edge;  //assume reference 3cm of edge
  return scaleFactor;
}

float HGCHEbackSignalScaler::computeEdge(const HGCScintillatorDetId& cellId)
{
  float radius = computeRadius(cellId);
  float circ = 2 * M_PI * radius;

  float edge(3.);
  if(cellId.type() == 0)
  {
    edge = circ / 360.; //1 degree
  }
  else
  {
    edge = circ / 288.; //1.25 degrees
  }

  if(verbose_)
  {
    std::cout << "HGCHEbackSignalScaler::computeEdge - Type, layer, edge, radius: "
              << cellId.type() << " "
              <<  cellId.layer() << " "
              << edge << " "
              << radius << std::endl;
  }

  return edge;
}

float HGCHEbackSignalScaler::computeRadius(const HGCScintillatorDetId& cellId)
{
  GlobalPoint global = hgcalGeom_->getPosition(cellId);
  float radius = sqrt( std::pow(global.x(), 2) + std::pow(global.y(), 2));
  return radius;
}




//--- the actual digitizer --------------------------------------------------------------------------------------------------
HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps)
{
  edm::ParameterSet cfg = ps.getParameter<edm::ParameterSet>("digiCfg");
  algo_        = cfg.getParameter<uint32_t>("algo");
  scaleByArea_ = cfg.getParameter<bool>("scaleByArea");
  scaleByDose_ = cfg.getParameter<bool>("scaleByDose");
  doseMapFile_ = cfg.getParameter<std::string>("doseMap");
  keV2MIP_     = cfg.getParameter<double>("keV2MIP");
  this->keV2fC_    = 1.0; //keV2MIP_; // hack for HEB
  noise_MIP_   = cfg.getParameter<edm::ParameterSet>("noise_MIP").getParameter<double>("value");
  nPEperMIP_   = cfg.getParameter<double>("nPEperMIP");
  nTotalPE_    = cfg.getParameter<double>("nTotalPE");
  xTalk_       = cfg.getParameter<double>("xTalk");
  sdPixels_    = cfg.getParameter<double>("sdPixels");
}

//
void HGCHEbackDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
				      const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
				      uint32_t digitizationType, CLHEP::HepRandomEngine* engine)
{
  switch(algo_)
  {
    case 0:
      runEmptyDigitizer(digiColl,simData,theGeom,validIds,engine);
    case 1:
      runCaliceLikeDigitizer(digiColl,simData,theGeom,validIds,engine);
    case 2:
      runRealisticDigitizer(digiColl,simData,theGeom,validIds,engine);
  }
}

void HGCHEbackDigitizer::runEmptyDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
					   const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
					   CLHEP::HepRandomEngine* engine)
{
  HGCSimHitData chargeColl, toa;
  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f); //accumulated energy
  zeroData.hit_info[1].fill(0.f); //time-of-flight

  for( const auto& id : validIds ) {

    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = ( simData.end() == it ? zeroData : it->second );
    addCellMetadata(cell,theGeom,id);

    for(size_t i=0; i<cell.hit_info[0].size(); ++i)
    {
      //convert total energy keV->MIP, since converted to keV in accumulator
      const float totalIniMIPs( cell.hit_info[0][i]*keV2MIP_ );

      //store
      chargeColl[i] = totalIniMIPs;
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame( id );
    this->myFEelectronics_->runShaper( newDataFrame, chargeColl, toa, 1, engine );

    //prepare the output
    this->updateOutput(digiColl,newDataFrame);
  }
}

void HGCHEbackDigitizer::runRealisticDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
  const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
  CLHEP::HepRandomEngine* engine)
  {
    //switch to true if you want to print some details
    constexpr bool debug(false);

    HGCSimHitData chargeColl, toa;
    // this represents a cell with no signal charge
    HGCCellInfo zeroData;
    zeroData.hit_info[0].fill(0.f); //accumulated energy
    zeroData.hit_info[1].fill(0.f); //time-of-flight

    // needed to compute the radiation and geometry scale factors
    HGCHEbackSignalScaler scal(theGeom, doseMapFile_);

    for( const auto& id : validIds ) {

      chargeColl.fill(0.f);
      toa.fill(0.f);
      HGCSimHitDataAccumulator::iterator it = simData.find(id);
      HGCCellInfo& cell = ( simData.end() == it ? zeroData : it->second );
      addCellMetadata(cell,theGeom,id);

      for(size_t i=0; i<cell.hit_info[0].size(); ++i)
      {
        //convert total energy keV->MIP, since converted to keV in accumulator
        float totalIniMIPs( cell.hit_info[0][i]*keV2MIP_ );

        //take into account the different size of the tiles
        if(scaleByArea_)
          totalIniMIPs *= scal.scaleByArea(id);
        //take into account the darkening of the scintillator
        if(scaleByDose_)
          totalIniMIPs *= scal.scaleByDose(id);

        //generate the number of photo-electrons from the energy deposit
        const uint32_t npeS = std::floor(CLHEP::RandPoissonQ::shoot(engine, totalIniMIPs * nPEperMIP_) + 0.5);

        //generate the noise associated to the dark current
        float meanN = std::pow(nPEperMIP_ * noise_MIP_, 2);
        const uint32_t npeN = std::floor(CLHEP::RandPoissonQ::shoot(engine, meanN) + 0.5);

        //total number of pe from signal + noise  (not subtracting pedestal)
        const uint32_t npe = npeS + npeN;

        //take into account SiPM saturation
        const float x = vdt::fast_expf( -((float)npe)/nTotalPE_ );
        uint32_t nPixel(0);
        if(xTalk_*x!=1)  nPixel = (uint32_t) std::max( nTotalPE_ * (1.f - x)/(1.f - xTalk_ * x), 0.f );

        //take into account the gain fluctuations of each pixel
        //const float nPixelTot = nPixel + sqrt(nPixel) * CLHEP::RandGaussQ::shoot(engine, 0., 0.05); //FDG: just a note for now, par to be defined

        //convert back to MIP without un-doing the saturation
        const float totalMIPs = nPixel / nPEperMIP_;

        if(debug && totalIniMIPs > 0)
        {
          std::cout << "npeS: " << npeS
          << " npeN: " << npeN
          << " npe: " << npe
          << " meanN: " << meanN
          << " noise_MIP_: " << noise_MIP_
          << " nPEperMIP_: " << nPEperMIP_
          << " nPixel: " << nPixel << std::endl;
          std::cout << "totalIniMIPs: " << totalIniMIPs << " totalMIPs: " << totalMIPs << std::endl;
        }

        //store
        chargeColl[i] = totalMIPs;
      }


      //init a new data frame and run shaper
      HGCalDataFrame newDataFrame( id );
      this->myFEelectronics_->runShaper( newDataFrame, chargeColl, toa, 1, engine );

      //prepare the output
      this->updateOutput(digiColl,newDataFrame);
    }
  }

//
void HGCHEbackDigitizer::runCaliceLikeDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
						const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
						CLHEP::HepRandomEngine* engine)
{
  //switch to true if you want to print some details
  constexpr bool debug(false);

  HGCSimHitData chargeColl, toa;

  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f); //accumulated energy
  zeroData.hit_info[1].fill(0.f); //time-of-flight

  for( const auto& id : validIds ) {
    chargeColl.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = ( simData.end() == it ? zeroData : it->second );
    addCellMetadata(cell,theGeom,id);

    for(size_t i=0; i<cell.hit_info[0].size(); ++i)
      {
	//convert total energy keV->MIP, since converted to keV in accumulator
	const float totalIniMIPs( cell.hit_info[0][i]*keV2MIP_ );
	//std::cout << "energy in MIP: " << std::scientific << totalIniMIPs << std::endl;

	  //generate random number of photon electrons
	  const uint32_t npe = std::floor(CLHEP::RandPoissonQ::shoot(engine,totalIniMIPs*nPEperMIP_));

	  //number of pixels
	  const float x = vdt::fast_expf( -((float)npe)/nTotalPE_ );
	  uint32_t nPixel(0);
	  if(xTalk_*x!=1) nPixel=(uint32_t) std::max( nTotalPE_*(1.f-x)/(1.f-xTalk_*x), 0.f );

	  //update signal
	  if(sdPixels_ != 0) nPixel = (uint32_t)std::max( CLHEP::RandGaussQ::shoot(engine,(double)nPixel,sdPixels_), 0. );

	  //convert to MIP again and saturate
          float totalMIPs(0.f), xtalk = 0.f;
          const float peDiff = nTotalPE_ - (float) nPixel;
          if (peDiff != 0.f) {
            xtalk = (nTotalPE_-xTalk_*((float)nPixel)) / peDiff;
            if( xtalk > 0.f && nPEperMIP_ != 0.f)
              totalMIPs = (nTotalPE_/nPEperMIP_)*vdt::fast_logf(xtalk);
          }

	  //add noise (in MIPs)
	  chargeColl[i] = totalMIPs;
      if(noise_MIP_ != 0) chargeColl[i] += std::max( CLHEP::RandGaussQ::shoot(engine,0.,noise_MIP_), 0. );
	  if(debug && cell.hit_info[0][i]>0)
	    std::cout << "[runCaliceLikeDigitizer] xtalk=" << xtalk << " En=" << cell.hit_info[0][i] << " keV -> " << totalIniMIPs << " raw-MIPs -> " << chargeColl[i] << " digi-MIPs" << std::endl;
	}

      //init a new data frame and run shaper
      HGCalDataFrame newDataFrame( id );
      this->myFEelectronics_->runShaper( newDataFrame, chargeColl, toa, 1, engine );

      //prepare the output
      this->updateOutput(digiColl,newDataFrame);
    }
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer()
{
}
