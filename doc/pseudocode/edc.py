def find_mismatches(): pass
C, control_checksums_row, control_checksums_col = 0
def notify_recmompute_checksums(): pass
def correct_error(): pass
checksum = 0


def f():

	errs_r = find_mismatches("row checksums", C, control_checksums_row)
	errs_c = find_mismatches("col checksums", C, control_checksums_col)

	if((len(errs_r) == 0) and (len(errs_c) == 0)):
		return "no errors"
	
	if((len(errs_r) > 1) and (len(errs_c) > 1)):
		return "uncorrectable errors"
	
	for err in (errs_r, errs_c):
		if(err is in checksum):
			notify_recmompute_checksums()
			continue

		correct_error()

	return "corrected errors"
