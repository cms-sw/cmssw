#! /bin/csh
if ($#argv < 1) then
    setenv RELEASE $CMSSW_VERSION
else
    setenv RELEASE $1
endif
#setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/ ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/TrackingRecHits

if (! -d $WWWDIRObj) mkdir $WWWDIRObj

mkdir $WWWDIRObj/Pixel

setenv WWWDIR $WWWDIRObj/Pixel

mkdir $WWWDIR/eps
mkdir $WWWDIR/gif

mv me*.eps.gz $WWWDIR/eps
mv summary*.eps.gz $WWWDIR/eps
mv me*.gif $WWWDIR/gif
mv summary*.gif $WWWDIR/gif

