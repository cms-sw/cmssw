#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/ ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/RecHits

if (! -d $WWWDIRObj) mkdir $WWWDIRObj

mkdir $WWWDIRObj/Strip

setenv WWWDIR $WWWDIRObj/Strip


mkdir $WWWDIR/eps
mkdir $WWWDIR/gif

mv *TIB*.eps.gz $WWWDIR/eps
mv *TID*.eps.gz $WWWDIR/eps
mv *TOB*.eps.gz $WWWDIR/eps
mv *TEC*.eps.gz $WWWDIR/eps
mv *TIB*.gif $WWWDIR/gif
mv *TID*.gif $WWWDIR/gif
mv *TOB*.gif $WWWDIR/gif
mv *TEC*.gif $WWWDIR/gif


