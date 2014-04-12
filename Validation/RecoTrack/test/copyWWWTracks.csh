#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/ ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/RecoTracks

if (! -d $WWWDIRObj) mkdir $WWWDIRObj

setenv WWWDIR $WWWDIRObj

mkdir $WWWDIR/eps
mkdir $WWWDIR/gif

mv ctf*.eps.gz $WWWDIR/eps
mv ctf*.gif $WWWDIR/gif


