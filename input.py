def read_file(title, format):

	# Read in filenames
	with open(title) as f:
	    files = f.readlines()
	files = [x.strip() for x in files]

	# Read file by file into the bloblist
	documents = []
	for (i, filename) in enumerate(files):
		file = open("../email_data/text_1000/" + filename + format)
		text = file.read()
		documents.append(text)

	return documents