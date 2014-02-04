#include "SimMuon/GEMDigitizer/interface/ME0PreRecoNoSmearModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <utility>
#include <map>


ME0PreRecoNoSmearModel::ME0PreRecoNoSmearModel(const edm::ParameterSet& config) 
  : ME0DigiPreRecoModel(config)
{
}

void 
ME0PreRecoNoSmearModel::simulateSignal(const ME0EtaPartition* roll,
				const edm::PSimHitContainer& simHits)
{

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;
    auto entry = hit.entryPoint();
    float x=entry.x();
    float y=entry.y(); 
    float ex=0.001;
    float ey=0.001;
    float corr=0.;
    float t = hit.timeOfFlight();
     // please keep hit time always 0 for this model
    ME0DigiPreReco digi(x,y,ex,ey,corr,t);
    digi_.insert(digi);
  }
}



