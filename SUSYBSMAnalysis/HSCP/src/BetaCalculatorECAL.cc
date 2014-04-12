#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalRecHitLess.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorECAL.h"

using namespace susybsm;

BetaCalculatorECAL::BetaCalculatorECAL(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC) :
  EBRecHitCollectionToken_(iC.consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitCollection"))),
  EERecHitCollectionToken_(iC.consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitCollection")))
{
   edm::ParameterSet trkParameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( trkParameters );
   trackAssociator_.useDefaultPropagator();

}


void BetaCalculatorECAL::addInfoToCandidate(HSCParticle& candidate, edm::Handle<reco::TrackCollection>& tracks, edm::Event& iEvent, const edm::EventSetup& iSetup,  HSCPCaloInfo& caloInfo) {
   bool setCalo = false;
   HSCPCaloInfo result;

   // EcalDetIdAssociator
   iSetup.get<DetIdAssociatorRecord>().get("EcalDetIdAssociator", ecalDetIdAssociator_);
   // Get the Bfield
   iSetup.get<IdealMagneticFieldRecord>().get(bField_);
   // Geometry
   iSetup.get<CaloGeometryRecord>().get(theCaloGeometry_);
   const CaloGeometry* theGeometry = theCaloGeometry_.product();
   // Topology
   edm::ESHandle<CaloTopology> pCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(pCaloTopology);
   const CaloTopology* theCaloTopology = pCaloTopology.product();
   // EcalRecHits
   edm::Handle<EBRecHitCollection> ebRecHits;
   iEvent.getByToken(EBRecHitCollectionToken_,ebRecHits);
   edm::Handle<EERecHitCollection> eeRecHits;
   iEvent.getByToken(EERecHitCollectionToken_,eeRecHits);

   // select the track
   reco::Track track;
   if(candidate.hasTrackRef())
     track = *(candidate.trackRef());
   else
     return; // in case there is no track ref, we can't do much

   // compute the track isolation
   result.trkIsoDr=100;
   for(reco::TrackCollection::const_iterator ndTrack = tracks->begin(); ndTrack != tracks->end(); ++ndTrack) {
       double dr=sqrt(pow((track.outerEta()-ndTrack->outerEta()),2)+pow((track.outerPhi()-ndTrack->outerPhi()),2));
       if(dr>0.00001 && dr<result.trkIsoDr) result.trkIsoDr=dr;
   }

   // use the track associator to propagate to the calo
   TrackDetMatchInfo info = trackAssociator_.associate( iEvent, iSetup,
                                                        trackAssociator_.getFreeTrajectoryState(iSetup, track),
                                                        parameters_ );

   // do a custom propagation through Ecal
   std::map<int,GlobalPoint> trackExitPositionMap; // rawId to exit position (subtracting cry center)
   std::map<int,float> trackCrossedXtalCurvedMap; // rawId to trackLength

   FreeTrajectoryState tkInnerState = trajectoryStateTransform::innerFreeState(track, &*bField_);
   // Build set of points in Ecal (necklace) using the propagator
   std::vector<SteppingHelixStateInfo> neckLace;
   neckLace = calcEcalDeposit(&tkInnerState,*ecalDetIdAssociator_);
   // Initialize variables to be filled by the track-length function
   double totalLengthCurved = 0.;
   GlobalPoint internalPointCurved(0., 0., 0.);
   GlobalPoint externalPointCurved(0., 0., 0.);
   if(neckLace.size() > 1)
   {
     getDetailedTrackLengthInXtals(trackExitPositionMap,
         trackCrossedXtalCurvedMap,
         totalLengthCurved,
         internalPointCurved,
         externalPointCurved,
         & (*theGeometry),
         & (*theCaloTopology),
         neckLace);
   }

   // Make weighted sum of times
   float sumWeightedTime = 0;
   float sumTimeErrorSqr = 0;
   float sumEnergy = 0;
   float sumTrackLength = 0;
   std::vector<EcalRecHit> crossedRecHits;
   EcalRecHitCollection::const_iterator thisHit;

   std::map<int,GlobalPoint>::const_iterator trackExitMapIt = trackExitPositionMap.begin();
   for(std::map<int,float>::const_iterator mapIt = trackCrossedXtalCurvedMap.begin();
       mapIt != trackCrossedXtalCurvedMap.end(); ++mapIt)
   {
     if(DetId(mapIt->first).subdetId()==EcalBarrel)
     {
       EBDetId ebDetId(mapIt->first);
       thisHit = ebRecHits->find(ebDetId);
       if(thisHit == ebRecHits->end())
       {
         //std::cout << "\t Could not find crossedEcal detId: " << ebDetId << " in EBRecHitCollection!" << std::endl;
         continue;
       }
       const EcalRecHit hit = *thisHit;
       // Cut out badly-reconstructed hits
       if(!hit.isTimeValid())
         continue;
       uint32_t rhFlag = hit.recoFlag();
       if((rhFlag != EcalRecHit::kGood) && (rhFlag != EcalRecHit::kOutOfTime) && (rhFlag != EcalRecHit::kPoorCalib))
         continue;

       float errorOnThis = hit.timeError();
       sumTrackLength+=mapIt->second;
       sumEnergy+=hit.energy();
       crossedRecHits.push_back(hit);
//       result.ecalSwissCrossKs.push_back(EcalSeverityLevelAlgo::spikeFromNeighbours(ebDetId,(*ebRecHits),0.2,EcalSeverityLevelAlgo::kSwissCross));
//       result.ecalE1OverE9s.push_back(EcalSeverityLevelAlgo::spikeFromNeighbours(ebDetId,(*ebRecHits),0.2,EcalSeverityLevelAlgo::kE1OverE9));
       result.ecalTrackLengths.push_back(mapIt->second);
       result.ecalTrackExitPositions.push_back(trackExitMapIt->second);
       result.ecalEnergies.push_back(hit.energy());
       result.ecalTimes.push_back(hit.time());
       result.ecalTimeErrors.push_back(hit.timeError());
       result.ecalOutOfTimeEnergies.push_back(hit.outOfTimeEnergy());
       result.ecalOutOfTimeChi2s.push_back(hit.outOfTimeChi2());
       result.ecalChi2s.push_back(hit.chi2());
       result.ecalDetIds.push_back(ebDetId);
       // SIC DEBUG
       //std::cout << " SIC DEBUG: time error on this crossed RecHit: " << errorOnThis << " energy of hit: "
       //  << hit.energy() << " time of hit: " << hit.time() << " trackLength: " << mapIt->second << std::endl;

       if(hit.isTimeErrorValid()) // use hit time for weighted time average
       {
         sumWeightedTime+=hit.time()/(errorOnThis*errorOnThis);
         sumTimeErrorSqr+=1/(errorOnThis*errorOnThis);
       }
     }
     trackExitMapIt++;
   }

   if(crossedRecHits.size() > 0)
   {
     setCalo = true;
     sort(crossedRecHits.begin(),crossedRecHits.end(),EcalRecHitLess());
     result.ecalCrossedEnergy = sumEnergy;
     result.ecalCrysCrossed = crossedRecHits.size();
     result.ecalDeDx = sumEnergy/sumTrackLength;
     // replace the below w/o trackassociator quantities?
     result.ecal3by3dir = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
     result.ecal5by5dir = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);

     if(sumTimeErrorSqr > 0)
     {
       result.ecalTime = sumWeightedTime/sumTimeErrorSqr;
       result.ecalTimeError = sqrt(1/sumTimeErrorSqr);
       DetId maxEnergyId = crossedRecHits.begin()->id();

       if(maxEnergyId != DetId()) // double check
       {
         // To get beta, we assume photon propagation time is about the same for muons and e/gamma
         // Since the typical path length is >> crystal length, this shouldn't be too bad
         GlobalPoint position = info.getPosition(maxEnergyId); // position of crystal center on front face
         double frontFaceR = sqrt(pow(position.x(),2)+pow(position.y(),2)+pow(position.z(),2));
         double muonShowerMax = frontFaceR+11.5; // assume muon "showerMax" is halfway into the crystal
         double gammaShowerMax = frontFaceR+6.23; // 7 X0 for e/gamma showerMax
         double speedOfLight = 29.979; // cm/ns
         result.ecalBeta = (muonShowerMax)/(result.ecalTime*speedOfLight+gammaShowerMax);
         result.ecalBetaError = (speedOfLight*muonShowerMax*result.ecalTimeError)/pow(speedOfLight*result.ecalTime+gammaShowerMax,2);
         result.ecalInvBetaError = speedOfLight*result.ecalTimeError/muonShowerMax;
       }
       // SIC debug
       //std::cout << "BetaCalcEcal: CrossedRecHits: " << crossedRecHits.size()
       //  << " ecalTime: " << result.ecalTime << " timeError: " << result.ecalTimeError
       //  << " ecalCrossedEnergy: " << result.ecalCrossedEnergy << " ecalBeta: " << result.ecalBeta
       //  << " ecalBetaError: " << result.ecalBetaError <<  " ecalDeDx (MeV/cm): " << 1000*result.ecalDeDx << std::endl;
     }
   }

   if(info.crossedHcalRecHits.size() > 0)
   {
     // HCAL (not ECAL) info
     result.hcalCrossedEnergy = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
     result.hoCrossedEnergy = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
     //maxEnergyId = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
     result.hcal3by3dir = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
     result.hcal5by5dir = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2);
   }

   if(setCalo)
     caloInfo = result;
}

std::vector<SteppingHelixStateInfo> BetaCalculatorECAL::calcEcalDeposit(const FreeTrajectoryState* tkInnerState,
            const DetIdAssociator& associator)
{
   // Set some parameters
   double minR = associator.volume().minR () ;
   double minZ = associator.volume().minZ () ;
   double maxR = associator.volume().maxR () ;
   double maxZ = associator.volume().maxZ () ;

   // Define the TrackOrigin (where the propagation starts)
   SteppingHelixStateInfo trackOrigin(*tkInnerState);

   // Define Propagator
   SteppingHelixPropagator* prop = new SteppingHelixPropagator (&*bField_, alongMomentum);
   prop -> setMaterialMode(false);
   prop -> applyRadX0Correction(true);

   return propagateThoughFromIP(trackOrigin,prop,associator.volume(), 500,0.1,minR,minZ,maxR,maxZ);
}

int BetaCalculatorECAL::getDetailedTrackLengthInXtals(std::map<int,GlobalPoint>& trackExitPositionMap,
   std::map<int,float>& trackCrossedXtalMap,
   double& totalLengthCurved,
   GlobalPoint& internalPointCurved,
   GlobalPoint& externalPointCurved,
   const CaloGeometry* theGeometry,
   const CaloTopology* theTopology,
   const std::vector<SteppingHelixStateInfo>& neckLace)
{
   GlobalPoint origin (0., 0., 0.);
   internalPointCurved = origin ;
   externalPointCurved = origin ;

   bool firstPoint = false;
   trackCrossedXtalMap.clear();

   const CaloSubdetectorGeometry* theBarrelSubdetGeometry = theGeometry->getSubdetectorGeometry(DetId::Ecal,1);
   const CaloSubdetectorGeometry* theEndcapSubdetGeometry = theGeometry->getSubdetectorGeometry(DetId::Ecal,2);

   for(std::vector<SteppingHelixStateInfo>::const_iterator itr = (neckLace.begin() + 1); itr != neckLace.end(); ++itr)
   {
     GlobalPoint probe_gp = (*itr).position();
     std::vector<DetId> surroundingMatrix;

     EBDetId closestBarrelDetIdToProbe = ((theBarrelSubdetGeometry -> getClosestCell(probe_gp)).rawId());
     EEDetId closestEndcapDetIdToProbe = ((theEndcapSubdetGeometry -> getClosestCell(probe_gp)).rawId());

     // check if the probe is inside the xtal
     if( (closestEndcapDetIdToProbe) && (theGeometry->getSubdetectorGeometry(closestEndcapDetIdToProbe)->
           getGeometry(closestEndcapDetIdToProbe)->inside(probe_gp)) )
     {
       double step = ((*itr).position() - (*(itr-1)).position()).mag();
       GlobalPoint point = itr->position();
       addStepToXtal(trackExitPositionMap, trackCrossedXtalMap, closestEndcapDetIdToProbe, step, point, theEndcapSubdetGeometry);
       totalLengthCurved += step;

       if (firstPoint == false)
       {
         internalPointCurved = probe_gp ;
         firstPoint = true ;
       }

       externalPointCurved = probe_gp ;
     }

     if( (closestBarrelDetIdToProbe) && (theGeometry->getSubdetectorGeometry(closestBarrelDetIdToProbe)->
           getGeometry(closestBarrelDetIdToProbe)->inside(probe_gp)) )
     {
       double step = ((*itr).position() - (*(itr-1)).position()).mag();
       GlobalPoint point = itr->position();
       addStepToXtal(trackExitPositionMap, trackCrossedXtalMap, closestBarrelDetIdToProbe, step, point, theBarrelSubdetGeometry);
       totalLengthCurved += step;

       if (firstPoint == false)
       {
         internalPointCurved = probe_gp ;
         firstPoint = true ;
       }

       externalPointCurved = probe_gp ;
     }
     else
     {
       // 3x3 matrix surrounding the probe
       surroundingMatrix = theTopology->getSubdetectorTopology(closestBarrelDetIdToProbe)->getWindow(closestBarrelDetIdToProbe,3,3);

       for( unsigned int k=0; k<surroundingMatrix.size(); ++k ) {
         if(theGeometry->getSubdetectorGeometry(surroundingMatrix.at(k))->getGeometry(surroundingMatrix.at(k))->inside(probe_gp))
         {
           double step = ((*itr).position() - (*(itr-1)).position()).mag();
           GlobalPoint point = itr->position();
           addStepToXtal(trackExitPositionMap, trackCrossedXtalMap, surroundingMatrix[k], step,
               point, theGeometry->getSubdetectorGeometry(surroundingMatrix.at(k)));
           totalLengthCurved += step;

           if (firstPoint == false)
           {
             internalPointCurved = probe_gp ;
             firstPoint = true ;
           }

           externalPointCurved = probe_gp ;
         }
       }

       // clear neighborhood matrix
       surroundingMatrix.clear();
     }
   }

   return 0;
}

void BetaCalculatorECAL::addStepToXtal(std::map<int,GlobalPoint>& trackExitPositionMap,
    std::map<int,float>& trackCrossedXtalMap,
    DetId aDetId,
    float step,
    GlobalPoint point,
    const CaloSubdetectorGeometry* theSubdetGeometry)
{

  const CaloCellGeometry *cell_p = theSubdetGeometry->getGeometry(aDetId);
  GlobalPoint p = (dynamic_cast <const TruncatedPyramid *> (cell_p))->getPosition(23);
  GlobalPoint diff(point.x()-p.x(),point.y()-p.y(),point.z()-p.z());

  std::map<int,GlobalPoint>::iterator xtal = trackExitPositionMap.find(aDetId.rawId());
  if (xtal!=trackExitPositionMap.end())
    ((*xtal).second)=diff;
  else
    trackExitPositionMap.insert(std::pair<int,GlobalPoint>(aDetId.rawId(),diff));

  std::map<int,float>::iterator xtal2 = trackCrossedXtalMap.find(aDetId.rawId());
  if (xtal2!= trackCrossedXtalMap.end())
    ((*xtal2).second)+=step;
  else
    trackCrossedXtalMap.insert(std::pair<int,float>(aDetId.rawId(),step));
}



