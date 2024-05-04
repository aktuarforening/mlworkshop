create-csv:
	cat ~/Downloads/freq.zip* > freq.zip
	zip -FF freq.zip --out freq-full.zip
	unzip freq-full.zip
	rm -rf freq*.zip

get-claims:
	cat ~/Downloads/claims.csv > claims.csv

serve-slides:
	jupyter nbconvert --to slides slides/${num}*.ipynb --post serve

