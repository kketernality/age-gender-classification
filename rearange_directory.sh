if [ ! -d data/child_male ]; then
    mkdir data/child_male 
fi
mv data/child/male/*.jpg data/child_male/
if [ ! -d data/child_female ]; then
    mkdir data/child_female
fi
mv data/child/female/*.jpg data/child_female/
if [ ! -d data/young_male ]; then
    mkdir data/young_male
fi
mv data/young/male/*.jpg data/young_male/
if [ ! -d data/young_female ]; then
    mkdir data/young_female
fi
mv data/young/female/*.jpg data/young_female/
if [ ! -d data/adult_male ]; then
    mkdir data/adult_male
fi
mv data/adult/male/*.jpg data/adult_male/
if [ ! -d data/adult_female ]; then
    mkdir data/adult_female
fi
mv data/adult/female/*.jpg data/adult_female/
if [ ! -d data/elder_male ]; then
    mkdir data/elder_male
fi
mv data/elder/male/*.jpg data/elder_male/
if [ ! -d data/elder_female ]; then
    mkdir data/elder_female
fi
mv data/elder/female/*.jpg data/elder_female/

rm -r data/child
rm -r data/young
rm -r data/adult
rm -r data/elder

