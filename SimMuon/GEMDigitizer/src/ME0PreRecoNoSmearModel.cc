#include "SimMuon/GEMDigitizer/interface/ME0PreRecoNoSmearModel.h"

#include <cmath>
#include <utility>
#include <map>


ME0PreRecoNoSmearModel::ME0PreRecoNoSmearModel(const edm::ParameterSet& config) 
  : ME0DigiPreRecoModel(config)
{
}

void 
ME0PreRecoNoSmearModel::simulateSignal(const ME0EtaPartition* roll,
				const edm::PSimHitContainer& simHits, CLHEP::HepRandomEngine* engine)
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
    int pdgid=hit.particleType();
     // please keep hit time always 0 for this model
    ME0DigiPreReco digi(x,y,ex,ey,corr,t,pdgid,1);
    digi_.insert(digi);
  }
}

void 
ME0PreRecoNoSmearModel::simulateNoise(const ME0EtaPartition* roll, CLHEP::HepRandomEngine* engine)
{
}



