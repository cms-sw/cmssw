#! /bin/csh 

## argv[1] == 1 
# copy reference files from afs and prepare cfg files

## argv[1] == 2
# submit cmsRun jobs


## argv[1] == 3 
# run the root macro

## argv[1] == 4 
# copy plots and root files to afs







######################
set Release=$CMSSW_VERSION
set Repository=/afs/cern.ch/cms/performance/tracker/activities/validation/misalignment
#set NewRepository=myLocalPath
#set Sequence=digi2track
#set Sequence=iterative
#set Sequence=newConfiguration
#set Sequence=re_tracking
set Sequence=only_validation
set scenarios=(STARTUP_V1 1PB_V1 10PB_V1 IDEAL)
#set samples=(RelValSingleMuPt1 RelValSingleMuPt10 RelValSingleMuPt100) 
#set samples=(RelValSinglePiPt1 RelValSinglePiPt10 RelValSinglePiPt100)
#set samples=(RelValTTbar RelValQCD_Pt_3000_3500 RelValQCD_Pt_80_120)
set samples=(RelValTTbar)
#set samples=(RelValSingleMuPt1 RelValSingleMuPt10 RelValSingleMuPt100 RelValSinglePiPt1 RelValSinglePiPt10 RelValSinglePiPt100 RelValTTbar RelValQCD_Pt_3000_3500 RelValQCD_Pt_80_120)
#set cfg = trackingPerformanceValidation13x.cfg
set cfg = trackingmisalignmentValidation.cfg
#####################


if($1 == 1) then
echo "you chose option 1 (Prepare cfg files)"

foreach sample($samples)
    foreach scenario($scenarios)
	if(! -d $Release) mkdir $Release
	if(! -d output_trees) mkdir output_trees

	if($sample == RelValZPrimeEEM4000) then
	    sed s/NEVENT/1000/g $cfg >! tmp1.cfg
	else if($sample == RelValQCD_Pt_3000_3500) then
	    sed s/NEVENT/5000/g $cfg >! tmp1.cfg
	else if($sample == RelValTTbar) then
	    sed s/NEVENT/5000/g $cfg >! tmp1.cfg
	else if($sample == RelValSingleMuPt1) then
	    sed s/NEVENT/-1/g $cfg >! tmp1.cfg
	else if($sample == RelValSingleMuPt10) then
	    sed s/NEVENT/-1/g $cfg >! tmp1.cfg
	else if($sample == RelValSingleMuPt100) then
	    sed s/NEVENT/-1/g $cfg >! tmp1.cfg
	else if($sample == RelValSinglePiPt1) then
	    sed s/NEVENT/-1/g $cfg >! tmp1.cfg
	else if($sample == RelValSinglePiPt10) then
	    sed s/NEVENT/-1/g $cfg >! tmp1.cfg
	else if($sample == RelValSinglePiPt100) then
	    sed s/NEVENT/-1/g $cfg >! tmp1.cfg
	else
	    sed s/NEVENT/5000/g $cfg >! tmp1.cfg
	endif

	sed s/SEQUENCE/$Sequence/g tmp1.cfg >! tmp2.cfg
	sed s/SAMPLE/$sample.$scenario/g tmp2.cfg >! tmp3.cfg
	sed s/SCENARIO/$scenario/g tmp3.cfg >! $sample.$scenario.cfg

	touch $sample.$scenario.cff
    end

end


else if($1 == 2) then
echo "you chose option 2 (Start validation)"

foreach sample($samples)
    foreach scenario($scenarios)
	eval `scramv1 run -csh`
	cmsRun $sample.$scenario.cfg >& ! $sample.$scenario.log &
    end
end

## to be modified....
else if($1 == 3) then
echo "you chose option 3 (Produce eps and gif plots)"
foreach sample($samples)

	sed s~SAMPLE~$sample~g  Draw_comparison.C >! tmp1.C
	sed s~SCENARIOS~"$scenarios"~g tmp1.C >! $sample.C
     root -b -q $sample.C > ! macro.$sample.log

    rm tmp*.C
end




else if($1 == 4) then
echo "you chose option 4 (Move files on web page)"
foreach sample( $samples)
     if ( ! -d $Repository/$Release)  mkdir $Repository/$Release
     if ( ! -d $Repository/$Release/$sample)  mkdir $Repository/$Release/$sample
     if ( ! -d $Repository/$Release/$sample/eps)  mkdir $Repository/$Release/$sample/eps
     if ( ! -d $Repository/$Release/$sample/gif)  mkdir $Repository/$Release/$sample/gif

    echo "copying pdf files for sample: " $sample
     gzip *$sample*.eps
     cp *$sample*.eps.gz $Repository/$Release/$sample/eps
     cp *$sample*.gif $Repository/$Release/$sample/gif/

    echo "copying root file for sample: " $sample
    cp val.$sample.*.root $Repository/$Release/$sample
     
    echo "copying cff file for sample: " $sample
    cp $sample.*.cff $Repository/$Release/$sample

    echo "copying cfg file for sample: " $sample
    cp $sample.*.cfg $Repository/$Release/$sample

    rm *.eps.gz
    rm *.gif
 end

else

    echo "you have to choose among option 1, option 2, option 3 and option 4"
endif


