import sys, re, os

selectionsList = []
timeSelection = "between(t,0,"
filename = inputFile = end = start = None

# read standard input once, line by line
for line in sys.stdin:
	# detect a start of silence, which is an end of our selection
	end = re.search(r"silence_start: (\d+\.?\d+)", line)
	# detect an end of silence, which is a start of our selection
	start = re.search(r"silence_end: (\d+\.?\d+)", line)

	if filename is None:
		# find the input filename
		filename = re.search(r"Input .+ from '(.+)':", line)
	else:
		if inputFile is None:
			inputFile = filename.group(1)

	if start is not None:
		timeSelection = "between(t," + start.group(1) + ","
	if end is not None:
		#timeSelection += end.group(1) + ")"
		timeSelection += str(float(end.group(1))+0.2) + ")"
		selectionsList.append(timeSelection)

# Note: silencedetect apparently handles properly files that start and/or end in silence
# so we don't need to check for that and complete filters with no start or no end
selectionFilter = "'" + "+".join(selectionsList) + "'"

vfilter = "-vf \"select=" + selectionFilter + ",setpts=N/FRAME_RATE/TB\""
afilter = "-af \"aselect=" + selectionFilter + ",asetpts=N/SR/TB\""

# output ffmpeg command
videos_dir = os.path.dirname(inputFile)
input_filename = os.path.basename(inputFile)
output_filename = f"outfile_{input_filename}"
output_file = os.path.join(videos_dir, output_filename)

print("ffmpeg -y -i", inputFile, vfilter, afilter, output_file)
