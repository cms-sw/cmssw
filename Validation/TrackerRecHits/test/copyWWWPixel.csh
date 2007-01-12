#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if (-e /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/ ) mkdir /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/RecHits

if (-e $WWWDIRObj) mkdir $WWWDIRObj

mkdir $WWWDIRObj/Pixel

setenv WWWDIR $WWWDIRObj/Pixel

mkdir $WWWDIR/eps
mkdir $WWWDIR/gif

mv *.eps $WWWDIR/eps
mv *.gif $WWWDIR/gif


