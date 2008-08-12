#! /bin/tcsh 

## argv[1] == 1 
# copy reference files from afs and prepare cfg files

## argv[1] == 2
# submit cmsRun jobs


## argv[1] == 3 
# run the root macro

## argv[1] == 4 
# copy plots and root files to afs







######################
set RefRelease=CMSSW_2_1_0_pre6
set NewRelease=$CMSSW_VERSION
set Algo=""
set Quality=""
#set Quality="highPurity"
set Tracks="generalTracks"
#set Tracks="cutsRecoTracks"
set GlobalTag = IDEAL_V5
set RefSelection=IDEAL_V2_out_of_the_box
set NewSelection=${GlobalTag}_out_of_the_box${Algo}${Quality}
set RefRepository=/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance
set NewRepository=/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance
#set NewRepository=myLocalPath
#set Sequence=digi2track
#set Sequence=iterative
#set Sequence=newConfiguration
#set Sequence=re_tracking
set Sequence=only_validation
#set samples=(RelValSingleMuPt1 RelValSingleMuPt10  RelValSingleMuPt100) 
#set samples=(RelValSingleMuPt1 RelValSingleMuPt10 RelValSingleMuPt100) 
set samples=(RelValSingleMuPt100)
#set samples=(RelValSinglePiPt1)
#set samples=(RelValTTbar)
#set samples=(RelValSingleMuPt1 RelValSingleMuPt10 RelValSingleMuPt100 RelValSinglePiPt1 RelValSinglePiPt10 RelValSinglePiPt100 RelValTTbar RelValQCD_Pt_3000_3500 RelValQCD_Pt_80_120)
#set cfg = trackingPerformanceValidation13x.cfg
set cfg = trackingPerformanceValidation_cfg.py
#set cfg2 = trackingPerformanceValidation_cfg.py
#####################
#echo $rootfiles
#exit 0

if($1 == 1) then
echo "you chose option 1"

foreach sample($samples)

    if(! -d $RefRelease) mkdir $RefRelease
    if(! -d output_trees) mkdir output_trees
    if(! -e $sample.cff) then 
    echo "Missing "$sample".cff, skipping this sample..."
    continue
    endif
    cp $RefRepository/$RefRelease/$RefSelection/$sample/val.$sample.root $RefRelease
#    cat cfg1 >! tmp0.cfg
    cat $sample.cff |grep -v maxEvents>! tmp0.cfg
    cat $cfg >> tmp0.cfg
    #cp $RefRepository/$RefRelease/$RefSelection/$sample/$sample.cff .

    if($sample == RelValZPrimeEEM4000) then
    sed s/NEVENT/1000/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValQCD_Pt_3000_3500) then
    sed s/NEVENT/5000/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValTTbar) then
    sed s/NEVENT/5000/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValSingleMuPt1) then
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValSingleMuPt10) then
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValSingleMuPt100) then
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValSinglePiPt1) then
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValSinglePiPt10) then
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    else if($sample == RelValSinglePiPt100) then
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    else
    sed s/NEVENT/-1/g tmp0.cfg >! tmp1.cfg
    endif
    sed s/GLOBALTAG/$GlobalTag/g tmp1.cfg >! tmp2.cfg
    sed s/SEQUENCE/$Sequence/g tmp2.cfg >! tmp3.cfg
    sed s/SAMPLE/$sample/g tmp3.cfg >!  tmp4.cfg
    sed s/ALGORITHM/$Algo/g tmp4.cfg >!  tmp5.cfg
    sed s/QUALITY/$Quality/g tmp5.cfg >!  tmp6.cfg
    sed s/TRACKS/$Tracks/g tmp6.cfg >!  tmp7.cfg
    sed -e "s/[ ]\+source/source/g" tmp7.cfg >!  $sample.py


#touch $sample.cff

end
rm tmp?.cfg

else if($1 == 2) then
echo "you chose option 2"

foreach sample($samples)

eval `scramv1 run -csh`
#cmsRun $sample.cfg >& ! $sample.log &
cmsRun $sample.py >& ! $sample.log &

end

else if($1 == 3) then
echo "you chose option 3"
foreach sample($samples)

    sed s~NEW_FILE~val.$sample.root~g macro/TrackValHistoPublisher.C >! tmp1.C
    sed s~REF_FILE~$RefRelease/val.$sample.root~g tmp1.C >! tmp2.C
    sed s~REF_LABEL~$sample~g tmp2.C >! tmp3.C
    sed s~NEW_LABEL~$sample~g tmp3.C >! tmp4.C
    sed s~REF_RELEASE~$RefRelease~g tmp4.C >! tmp5.C
    sed s~NEW_RELEASE~$NewRelease~g tmp5.C >! tmp6.C
    sed s~REFSELECTION~$RefSelection~g tmp6.C >! tmp7.C
    sed s~NEWSELECTION~$NewSelection~g tmp7.C >! tmp8.C
    sed s~TrackValHistoPublisher~$sample~g tmp8.C >! $sample.C

    root -b -q $sample.C > ! macro.$sample.log
    #$CMSSW_RELEASE_BASE/test/slc4_ia32_gcc345/hltTimingSummary -t 10000 -b100 -o $sample.timing -s -i output.$sample.root
    if ( ! -d $NewRepository/$NewRelease)  mkdir $NewRepository/$NewRelease
    if ( ! -d $NewRepository/$NewRelease/$NewSelection) mkdir $NewRepository/$NewRelease/$NewSelection
    if ( ! -d $NewRepository/$NewRelease/$NewSelection/$sample) mkdir $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying pdf files for sample: " $sample
    cp *.pdf $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying root file for sample: " $sample
    cp val.$sample.root $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying cff file for sample: " $sample
    cp $sample.cff $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying cfg file for sample: " $sample
    #cp $sample.cfg $NewRepository/$NewRelease/$NewSelection/$sample
    cp $sample.py $NewRepository/$NewRelease/$NewSelection/$sample
    rm tmp*.C
    rm *.pdf

end




else if($1 == 4) then
echo "you chose option 4"
foreach sample( $samples)
    if (  -f $sample.cfg ) rm $sample.cfg
    if (  -f $sample.cff ) rm $sample.cff  
    if (  -f $sample.C ) rm $sample.C
    if (  -f $sample.log ) rm $sample.log
    if (  -f val.$sample.root ) rm val.$sample.root
    if (  -f macro.$sample.log ) rm macro.$sample.log
end

else

    echo "you have to choose among option 1, option 2, option 3 and option 4"
endif


