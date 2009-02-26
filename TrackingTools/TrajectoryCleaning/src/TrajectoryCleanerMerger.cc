#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerMerger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <map>
#include <vector>

using namespace std;

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <fstream>

/*****************************************************************************/
class HitComparator
{
  public:
    bool operator() (const TransientTrackingRecHit* ta,
                     const TransientTrackingRecHit* tb) const
    {
      const TrackingRecHit* a = ta->hit();
      const TrackingRecHit* b = tb->hit();

      if(getId(a) < getId(b)) return true;
      if(getId(b) < getId(a)) return false;

      if(a->geographicalId() < b->geographicalId()) return true;
      if(b->geographicalId() < a->geographicalId()) return false;

      const SiPixelRecHit* a_ = dynamic_cast<const SiPixelRecHit*>(a);
      if(a_ != 0)
      {
        const SiPixelRecHit* b_ = dynamic_cast<const SiPixelRecHit*>(b);
        return less(a_, b_);
      }
      else
      {
        const SiStripMatchedRecHit2D* a_ =
          dynamic_cast<const SiStripMatchedRecHit2D*>(a);

        if(a_ != 0)
        {
          const SiStripMatchedRecHit2D* b_ =
            dynamic_cast<const SiStripMatchedRecHit2D*>(b);
          return less(a_, b_);
        }
        else
        {
          const SiStripRecHit2D* a_ =
            dynamic_cast<const SiStripRecHit2D*>(a);

          if(a_ != 0)
          {
            const SiStripRecHit2D* b_ =
              dynamic_cast<const SiStripRecHit2D*>(b);
            return less(a_, b_);
          }
          else 
          {
            const ProjectedSiStripRecHit2D* a_ =
              dynamic_cast<const ProjectedSiStripRecHit2D*>(a); 

//cerr << " comp proj" << endl;

            if(a_ != 0)
            {
              const ProjectedSiStripRecHit2D* b_ =
                dynamic_cast<const ProjectedSiStripRecHit2D*>(b);

              return less(&(a_->originalHit()), &(b_->originalHit()));
            }
            else
              return false;
          }
        }
      }
    }

  private:
    int getId(const TrackingRecHit* a) const
    {
      if(dynamic_cast<const SiPixelRecHit*>(a)            != 0) return 0;
      if(dynamic_cast<const SiStripRecHit2D*>(a)          != 0) return 1;
      if(dynamic_cast<const SiStripMatchedRecHit2D*>(a)   != 0) return 2;
      if(dynamic_cast<const ProjectedSiStripRecHit2D*>(a) != 0) return 3;
      return -1;
    }

    bool less(const SiPixelRecHit* a,
              const SiPixelRecHit* b) const
    {
//cerr << " comp pixel" << endl;
      return a->cluster() < b->cluster();
    }

    bool less(const SiStripRecHit2D* a,
              const SiStripRecHit2D *b) const
    {
//cerr << " comp strip" << endl;
      return a->cluster() < b->cluster();
    }

    bool less(const SiStripMatchedRecHit2D* a,
              const SiStripMatchedRecHit2D *b) const
    {
//cerr << " comp matched strip" << endl;
      if(less(a->monoHit(), b->monoHit())) return true;
      if(less(b->monoHit(), a->monoHit())) return false;

      if(less(a->stereoHit(), b->stereoHit())) return true;
      return false;
    }
};

/*****************************************************************************/
void TrajectoryCleanerMerger::clean( TrajectoryPointerContainer&) const
{
}

/*****************************************************************************/
void TrajectoryCleanerMerger::reOrderMeasurements(Trajectory& traj)const
{
  vector<TrajectoryMeasurement> meas_ = traj.measurements();
  vector<TrajectoryMeasurement> meas;

  for(vector<TrajectoryMeasurement>::iterator
       im = meas_.begin();
       im!= meas_.end(); im++)
    if(im->recHit()->isValid())
       meas.push_back(*im);

  bool changed;

  do
  {
    changed = false;

    for(vector<TrajectoryMeasurement>::iterator im = meas.begin();
                                                im!= meas.end()-1; im++)
    if(    (*im).recHit()->globalPosition().mag2() >
       (*(im+1)).recHit()->globalPosition().mag2() + 1e-6)
    {
      swap(*im,*(im+1));
      changed = true;
    }
  }
  while(changed);

  for(unsigned int i = 0 ; i < meas.size(); i++)
     traj.pop();

  for(vector<TrajectoryMeasurement>::iterator im = meas.begin();
                                              im!= meas.end(); im++)
    traj.push(*im);
}
/*****************************************************************************/
bool TrajectoryCleanerMerger::sameSeed  (const TrajectorySeed & s1,   const TrajectorySeed & s2)const
{
  if(s1.nHits() != s2.nHits()) return false;

  TrajectorySeed::range r1 = s1.recHits();
  TrajectorySeed::range r2 = s2.recHits();

  TrajectorySeed::const_iterator h1 = r1.first;
  TrajectorySeed::const_iterator h2 = r2.first;

  do
  {
    if(!(h1->sharesInput(&(*h2),TrackingRecHit::all)))
      return false;

    h1++; h2++;
  }
  while(h1 != s1.recHits().second && 
        h2 != s2.recHits().second);

  return true;
}

/*****************************************************************************/
int TrajectoryCleanerMerger::getLayer(const DetId & id)const
{
  // PXB layer, ladder -> (layer - 1)<<2 + (ladder-1)%2
  // PXF disk , panel
  // TIB layer, module 
  // TOB layer, module
  // TID wheel, ring
  // TEC wheel, ring

  if(id.subdetId() == (uint) PixelSubdetector::PixelBarrel)
  { PXBDetId pid(id); return (100 * id.subdetId()+ (pid.layer() - 1)<<1 + (pid.ladder() - 1)%2); }

  if(id.subdetId() == (uint) PixelSubdetector::PixelEndcap)
  { PXFDetId pid(id); return (100 * id.subdetId()+ (pid.disk()  - 1)<<1 + (pid.panel()  - 1)%2); }

  if(id.subdetId() == StripSubdetector::TIB)
  { TIBDetId pid(id); return (100 * id.subdetId()+ (pid.layer() - 1)<<1 + (pid.module() - 1)%2); }
  if(id.subdetId() == StripSubdetector::TOB)
  { TOBDetId pid(id); return (100 * id.subdetId()+ (pid.layer() - 1)<<1 + (pid.module() - 1)%2); }

  if(id.subdetId() == StripSubdetector::TID)
  { TIDDetId pid(id); return (100 * id.subdetId()+ (pid.wheel() - 1)<<1 + (pid.ring()   - 1)%2); }
  if(id.subdetId() == StripSubdetector::TEC)
  { TECDetId pid(id); return (100 * id.subdetId()+ (pid.wheel() - 1)<<1 + (pid.ring()   - 1)%2); }

  return 0;
}

/***************************************************************************/

void TrajectoryCleanerMerger::clean
  (TrajectoryContainer& trajs) const 
{
  if(trajs.size() == 0) return;

  // Fill the rechit map
  typedef map<const TransientTrackingRecHit*,
              vector<unsigned int>, HitComparator> RecHitMap; 
  RecHitMap recHitMap;

  vector<bool> keep(trajs.size(),true);

  for(unsigned int i = 0; i < trajs.size(); i++) 
  {
    vector<TrajectoryMeasurement> meas = trajs[i].measurements();

    for(vector<TrajectoryMeasurement>::iterator im = meas.begin();
                                                im!= meas.end(); im++)
      if(im->recHit()->isValid())
      {
        const TransientTrackingRecHit* recHit = &(*(im->recHit()));
        if(recHit->isValid())
          recHitMap[recHit].push_back(i);
      }
  }

  // Look at each track
  typedef map<unsigned int,int,less<unsigned int> > TrajMap;

  for(unsigned int i = 0; i < trajs.size(); i++)
  if(keep[i])
  {  
    TrajMap trajMap;
    vector<DetId> detIds;
    vector<int> detLayers;

    // Go trough all rechits of this track
    vector<TrajectoryMeasurement> meas = trajs[i].measurements();
    for(vector<TrajectoryMeasurement>::iterator im = meas.begin();
                                                im!= meas.end(); im++)
    {
      if(im->recHit()->isValid())
      {
        // Get trajs sharing this rechit
        const TransientTrackingRecHit* recHit = &(*(im->recHit()));
        const vector<unsigned int>& sharing(recHitMap[recHit]);

        for(vector<unsigned int>::const_iterator j = sharing.begin(); 
                                                 j!= sharing.end(); j++)
          if(i < *j) trajMap[*j]++;

        // Fill detLayers vector
        detIds.push_back(recHit->geographicalId());
        detLayers.push_back(getLayer(recHit->geographicalId()));
      }
    }

    // Check for trajs with shared rechits
    for(TrajMap::iterator sharing = trajMap.begin();
                          sharing!= trajMap.end(); sharing++)
    {
      unsigned int j = (*sharing).first;
      if(!keep[i] || !keep[j]) continue;

      // More than 50% shared
      if((*sharing).second > min(trajs[i].foundHits(),
                                 trajs[j].foundHits())/2)
      {
        if( sameSeed(trajs[i].seed(), trajs[j].seed()) )
        {
        bool hasCommonLayer = false;

/*
        vector<TrajectoryMeasurement> measi = trajs[i].measurements();
        vector<TrajectoryMeasurement> measj = trajs[j].measurements();
        for(vector<TrajectoryMeasurement>::iterator
              tmj = measj.begin(); tmj!= measj.end(); tmj++)
            if(find(measi.begin(), measi.end(), tmj) == measi.end())
            if(find(detLayers.begin(),detLayers.end(),
                    getLayer(tmj->recHit()->geographicalId()))
                                   != detLayers.end())
             hasCommonLayer = true;
*/

        if(hasCommonLayer == false)
        { // merge tracks, add separate hits of the second to the first one
        vector<TrajectoryMeasurement> measj = trajs[j].measurements();
        for(vector<TrajectoryMeasurement>::iterator
             tmj = measj.begin(); tmj!= measj.end(); tmj++)
        if(tmj->recHit()->isValid())
        {
          bool match = false;

          vector<TrajectoryMeasurement> measi = trajs[i].measurements();
          for(vector<TrajectoryMeasurement>::iterator
             tmi = measi.begin(); tmi!= measi.end(); tmi++)
          if(tmi->recHit()->isValid())
            if(!HitComparator()(&(*(tmi->recHit())),
                                &(*(tmj->recHit()))) &&
               !HitComparator()(&(*(tmj->recHit())),
                                &(*(tmi->recHit()))))
            { match = true ; break; }

          if(!match)
            trajs[i].push(*tmj);
        }

        // Remove second track
        keep[j] = false;
        }
        else
        {
          // remove track with higher impact / chi2
          if(trajs[i].chiSquared() < trajs[j].chiSquared())
            keep[j] = false;
          else
            keep[i] = false;
        }
        }
      }
    } 
  }

  // Final copy
  int ok = 0;
  for(unsigned int i = 0; i < trajs.size(); i++)
    if(keep[i])
    {
      reOrderMeasurements(trajs[i]);
      ok++;
    }
    else
      trajs[i].invalidate();

  cerr << " [TrajecCleaner] cleaned trajs : " << ok << "/" << trajs.size() <<
" (with " << trajs[0].measurements().size() << "/" << recHitMap.size() << " hits)" << endl;
}

