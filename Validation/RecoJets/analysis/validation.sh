#!/bin/sh 

echo "This script produces all the histograms and plots..."
echo -e "Enter the REFERNCE.root file, the NEW.root file, the directory name/histogram title and the normalization ([REFERNCE.root] [NEW.root] [title] [normalization]): "
read reference new title norm

if [ -n "$reference" ]; then
    if [ -n "$new" ]; then
	if [ -n "$title" ]; then 
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

	    TAG=(CaloJetTask_iterativeCone5CaloJets CaloJetTask_L2L3CorJetIcone5 CaloJetTask_JetPlusTrackZSPCorJetIcone5 PFJetTask_iterativeCone5PFJets CaloJetTask_kt4CaloJets CaloJetTask_kt6CaloJets CaloJetTask_sisCone5CaloJets CaloJetTask_sisCone7CaloJets)
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

		if [ -z $norm ]; then
		../../../../test/slc4_ia32_gcc345/compareHists $new $reference ${TAG[${i}]} $title
		fi

		if [ $norm = "y" ]; then
		../../../../test/slc4_ia32_gcc345/compareHists $new $reference ${TAG[${i}]} $title y
		fi

		bash make_thumbnails.sh *.gif
		mv *.gif $title/${FOLDER[${i}]}
		mv *.jpg $title/${FOLDER[${i}]}
		cp html/spacer.gif $title/${FOLDER[${i}]}

		if [ ${FOLDER[${i}]} = ${FOLDER[3]} ]; then
		    cp html/${FOLDER[${i}]}.html $title/${FOLDER[${i}]}
		fi

		if [ ${FOLDER[${i}]} =  ${FOLDER[2]} ]; then
		    cp html/${FOLDER[${i}]}.html $title/${FOLDER[${i}]}
		else
		    cp html/htmlTemplate.html $title/${FOLDER[${i}]}
		fi

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
    cp $new       $title/new.root
    cp $reference $title/ref.root
else
    echo ".html files not copied!"
fi


