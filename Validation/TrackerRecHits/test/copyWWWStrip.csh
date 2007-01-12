#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/ ) mkdir /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/RecHits

if (! -d $WWWDIRObj) mkdir $WWWDIRObj

mkdir $WWWDIRObj/Strip

setenv WWWDIR $WWWDIRObj/Strip


mkdir $WWWDIR/eps
mkdir $WWWDIR/gif

mv *.eps $WWWDIR/eps
mv *.gif $WWWDIR/gif


