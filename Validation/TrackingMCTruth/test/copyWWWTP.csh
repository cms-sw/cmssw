#! /bin/csh
if ($#argv < 1) then
    setenv RELEASE $CMSSW_VERSION
else
    setenv RELEASE $1
endif
#setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/ ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/TrackingParticles

if (! -d $WWWDIRObj) mkdir $WWWDIRObj

setenv WWWDIR /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/TrackingParticles

mkdir $WWWDIR/eps

mkdir $WWWDIR/gif
echo "...Copying..."

mv *.eps.gz $WWWDIR/eps

mv *.gif $WWWDIR/gif

echo "...Done..."
