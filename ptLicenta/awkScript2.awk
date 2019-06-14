BEGIN{
	total = 0
	howMany = 10
	top = 5
}

{
	if(NR >= 2) {
			total += 1
			trackerCnt[$2] += 1
			trackersBestConfigs[$2][trackerCnt[$2]] = $18
			if(trackerCnt[$2] <= top) {
				print NR - 1, "&", $2, "&", $4, "&", $6, "&", $8, "&", $10, "&", $12, "&", $14, "&", $16, "&", $18, "\\\\"
			}
			trackerTime[$2] += $18
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
		print key, "&", mean, "\\\\"
	}
}
