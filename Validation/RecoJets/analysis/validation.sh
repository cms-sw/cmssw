#!/bin/sh 

echo "This script produces all the histograms and plots..."
echo -e "Enter the REFERNCE.root file, the NEW.root file and the directory name/histogram title([REFERNCE.root] [NEW.root] [title]): "
read reference new title

if [ -n "$reference" ]; then
    if [ -n "$new" ]; then
	if [ -n "$title"]; then
	    echo "processing "$reference" and "$new"."
	else 
	    echo "Input not correct!!"
	fi
    else 
	echo "Input not correct!!"
    fi
else
    echo "Input not correct!!"
fi

if [ -n $title ]; then
    if [ -e $reference ]; then
	if [ -e $new ]; then

	    TAG=(CaloJetTask_iterativeCone5CaloJets CaloJetTask_L2L3CorJet CaloJetTask_JetPlusTrackZSPCorJetIcone5 PFJetTask_iterativeCone5PFJets CaloJetTask_kt4CaloJets CaloJetTask_kt6CaloJets CaloJetTask_sisCone5CaloJets CaloJetTask_sisCone7CaloJets)
	    FOLDER=(Icone5 CorrIcone5 JPT PFlow kt4 kt6 Siscone5 Siscone7)
	    ntag=${#TAG[@]}
	    echo "Number of module tags: " $ntag
	    for (( i=0;i<$ntag;i++ )); do
		echo "tag: " ${TAG[${i}]}

		mkdir $title
		mkdir $title/${FOLDER[${i}]}
		mkdir $title/${FOLDER[${i}]}/NewData
		mkdir $title/${FOLDER[${i}]}/RefData	    
		echo "folders are created"

		../../../../test/slc4_ia32_gcc345/compareHists $new $reference ${TAG[${i}]} $title y
		mv *.gif $title/${FOLDER[${i}]}
		cp html/${FOLDER[${i}]}.html $title/${FOLDER[${i}]}

		../../../../test/slc4_ia32_gcc345/fixed_plotHists $new ${TAG[${i}]}
		mv *.gif $title/${FOLDER[${i}]}/NewData

		../../../../test/slc4_ia32_gcc345/fixed_plotHists $reference ${TAG[${i}]}
		mv *.gif $title/${FOLDER[${i}]}/RefData
	    done

	else
	    echo $new "not found!"
	fi
    else
	echo $refernce "not found!"
    fi
    
else
    echo "Enter a title!"
fi

if [ -d $title ]; then
    cp html/index.html $title
else
    echo ".html files not copied!"
fi


