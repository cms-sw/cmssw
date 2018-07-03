#!/bin/tcsh

#Check for correct number of arguments
if ($#argv<4) then
    echo "Script needs 4 input variables"
    exit
endif

set VAL_FILE=$1
set REF_FILE=$2
set VAL_VERS=$3
set REF_VERS=$4

#Go to HGCalValidation macro directory
#cd $CMSSW_BASE/src/Validation/HGCalValidation/macro


#Create base directory and top directories

#Check if base directory already exists
if ! (-d ${VAL_VERS}_vs_${REF_VERS}) then
    echo "No base directory exist, so creating one.."
    mkdir ${VAL_VERS}_vs_${REF_VERS}
endif

cd ${VAL_VERS}_vs_${REF_VERS}


if ("$VAL_FILE" =~ *"TTbar"*) then
    echo "Making TTbar Plots"
    set PUflag = ""
    #ttbar
    if ! (-d TTbar) then
	echo "No TTbar directory exist, so creating one.."
	mkdir TTbar    
    endif
    cd TTbar
    if("$VAL_FILE" =~ *"NoPU"*) then
	if ! (-d NoPU) then
	    echo "No TTbar directory exist, so creating one.."
	    mkdir NoPU
	endif
	set PUflag = "NoPU"
    endif
    if("$VAL_FILE" =~ *"PU140"*) then
	if ! (-d PU140) then
	    echo "No TTbar directory exist, so creating one.."
	    mkdir PU140
	endif
	set PUflag = "PU140"
    endif
    if("$VAL_FILE" =~ *"PU200"*) then
	if ! (-d PU200) then
	    echo "No TTbar directory exist, so creating one.."
	    mkdir PU200
	endif
	set PUflag = "PU200"
    endif
    echo "PU: " $PUflag
    cd ../../
    root -b -q 'PlotHGChitCalibration.C("'${VAL_FILE}'","'${REF_FILE}'","'${VAL_VERS}'", "'${REF_VERS}'")'
    if("$PUflag" =~ "NoPU") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/TTbar/NoPU/
    else if("$PUflag" =~ "PU140") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/TTbar/PU140/
    else
    mv *png ${VAL_VERS}_vs_${REF_VERS}/TTbar/PU200/
    endif
endif

if ("$VAL_FILE" =~ *"DoubleGamma"*) then
    echo " Making DoubleGamma Plots"
    set PUflag = ""
    #ttbar
    if ! (-d DoubleGamma) then
	echo "No DoubleGamma directory exist, so creating one.."
	mkdir DoubleGamma    
    endif
    cd DoubleGamma
    if("$VAL_FILE" =~ *"NoPU"*) then
	if ! (-d NoPU) then
	    echo "No DoubleGamma directory exist, so creating one.."
	    mkdir NoPU
	endif
	set PUflag = "NoPU"
    endif
    if("$VAL_FILE" =~ *"PU140"*) then
	if ! (-d PU140) then
	    echo "No DoubleGamma directory exist, so creating one.."
	    mkdir PU140
	endif
	set PUflag = "PU140"
    endif
    if("$VAL_FILE" =~ *"PU200"*) then
	if ! (-d PU200) then
	    echo "No DoubleGamma directory exist, so creating one.."
	    mkdir PU200
	endif
	set PUflag = "PU200"
    endif
    echo "PU: " $PUflag
    cd ../../
    root -b -q 'PlotHGChitCalibration.C("'${VAL_FILE}'","'${REF_FILE}'","'${VAL_VERS}'", "'${REF_VERS}'")'
    if("$PUflag" =~ "NoPU") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleGamma/NoPU/
    else if("$PUflag" =~ "PU140") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleGamma/PU140/
    else
    mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleGamma/PU200/
    endif
endif

if ("$VAL_FILE" =~ *"DoubleEle"*) then
    echo " Making DoubleEle Plots"
    set PUflag = ""
    #ttbar
    if ! (-d DoubleEle) then
	echo "No DoubleEle directory exist, so creating one.."
	mkdir DoubleEle    
    endif
    cd DoubleEle
    if("$VAL_FILE" =~ *"NoPU"*) then
	if ! (-d NoPU) then
	    echo "No DoubleEle directory exist, so creating one.."
	    mkdir NoPU
	endif
	set PUflag = "NoPU"
    endif
    if("$VAL_FILE" =~ *"PU140"*) then
	if ! (-d PU140) then
	    echo "No DoubleEle directory exist, so creating one.."
	    mkdir PU140
	endif
	set PUflag = "PU140"
    endif
    if("$VAL_FILE" =~ *"PU200"*) then
	if ! (-d PU200) then
	    echo "No DoubleEle directory exist, so creating one.."
	    mkdir PU200
	endif
	set PUflag = "PU200"
    endif
    echo "PU: " $PUflag
    cd ../../
    root -b -q 'PlotHGChitCalibration.C("'${VAL_FILE}'","'${REF_FILE}'","'${VAL_VERS}'", "'${REF_VERS}'")'
    if("$PUflag" =~ "NoPU") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleEle/NoPU/
    else if("$PUflag" =~ "PU140") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleEle/PU140/
    else
    mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleEle/PU200/
    endif
endif

if ("$VAL_FILE" =~ *"DoubleMu"*) then
    echo " Making DoubleMu Plots"
    set PUflag = ""
    #ttbar
    if ! (-d DoubleMu) then
	echo "No DoubleMu directory exist, so creating one.."
	mkdir DoubleMu    
    endif
    cd DoubleMu
    if("$VAL_FILE" =~ *"NoPU"*) then
	if ! (-d NoPU) then
	    echo "No DoubleMu directory exist, so creating one.."
	    mkdir NoPU
	endif
	set PUflag = "NoPU"
    endif
    if("$VAL_FILE" =~ *"PU140"*) then
	if ! (-d PU140) then
	    echo "No DoubleMu directory exist, so creating one.."
	    mkdir PU140
	endif
	set PUflag = "PU140"
    endif
    if("$VAL_FILE" =~ *"PU200"*) then
	if ! (-d PU200) then
	    echo "No DoubleMu directory exist, so creating one.."
	    mkdir PU200
	endif
	set PUflag = "PU200"
    endif
    echo "PU: " $PUflag
    cd ../../
    root -b -q 'PlotHGChitCalibration.C("'${VAL_FILE}'","'${REF_FILE}'","'${VAL_VERS}'", "'${REF_VERS}'")'
    if("$PUflag" =~ "NoPU") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleMu/NoPU/
    else if("$PUflag" =~ "PU140") then
	mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleMu/PU140/
    else
    mv *png ${VAL_VERS}_vs_${REF_VERS}/DoubleMu/PU200/
    endif
endif
