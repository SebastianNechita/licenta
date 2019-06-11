BEGIN{
	total = 0
	howMany = 10
	top = 5 
}

{
	if ($14 > 0.052407) {
		total += 1
		trackerCnt[$2] += 1
		trackersBestConfigs[$2][trackerCnt[$2]] = $14
		if(trackerCnt[$2] <= top) {
			print $0
		}
		trackerTime[$2] += $16
	}
}

END {
	print "Total: ", total
	for (key in trackerCnt) {
		mean = 0
		for(i = 1 ; i <= howMany ; i++) {
			mean += trackersBestConfigs[key][i]
		}
		mean /= howMany
		print "Tracker: " key,"has", trackerCnt[key], "configurations;",  "Mean mAP(over the top", howMany, "configurations) =", mean
	}
}
