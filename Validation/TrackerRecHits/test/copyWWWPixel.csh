#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/ ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/RecHits

if (! -d $WWWDIRObj) mkdir $WWWDIRObj

mkdir $WWWDIRObj/Pixel

setenv WWWDIR $WWWDIRObj/Pixel

mkdir $WWWDIR/eps
mkdir $WWWDIR/gif

mv Clust_*.eps.gz $WWWDIR/eps
mv RecHit_*.eps.gz $WWWDIR/eps
mv Clust_*.gif $WWWDIR/gif
mv RecHit_*.gif $WWWDIR/gif


