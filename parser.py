import os
import os.path
import re


HALIDE_MERGE = 'a_copy\[i\].(?:min|max)'
HALIDE_INTERSECT = 'halide_intersect\[i\].(?:min|max)'
HALIDE_ENCLOSES = 'halide_encloses\[i\].(?:min|max)'
HALIDE_EMPTY = 'halide_empty\[i\].(?:min|max)'
NFM_MERGE = 'a\[i\].(?:min|max)'
NFM_INTERSECT = 'nfm_intersect\[i\].(?:min|max)'
NFM_ENCLOSES = 'nfm_encloses\[i\].(?:min|max)'
NFM_EMPTY = 'nfm_empty\[i\].(?:min|max)'

name_to_regex_pattern = [
	('merge', HALIDE_MERGE, NFM_MERGE),
	('intersect', HALIDE_INTERSECT, NFM_INTERSECT),
	('encloses', HALIDE_ENCLOSES, NFM_ENCLOSES),
	('empty', HALIDE_EMPTY, NFM_EMPTY)
]

name_to_regex = [(name, re.compile(r'%s\s+\((\d+)\)' % halide), re.compile(r'%s\s+\((\d+)\)' % nfm)) \
					for name, halide, nfm in name_to_regex_pattern]


ENDS_WITH = "debug.txt"
#ENDS_WITH = "distrib.txt"

debug_filenames = []
for dirpath, dirnames, filenames in os.walk("./apps/"):
	for filename in [f for f in filenames if f.endswith(ENDS_WITH)]:
		name = os.path.join(dirpath, filename)
		debug_filenames.append(name)

results = {}
for in_filename in debug_filenames:
	results_temp = {}
	with open(in_filename, "r") as infile:#, open(out_filename, "w") as outfile:
		for line in infile:
			for name, regex_halide, regex_nfm in name_to_regex:
				halide = regex_halide.findall(line)
				nfm = regex_nfm.findall(line)
				temp = results_temp.get(name, None)
				if temp == None:
					results_temp[name] = (list(map(int, halide)), list(map(int, nfm)))
				else:
					temp[0].extend(list(map(int, halide)))
					temp[1].extend(list(map(int, nfm)))
	results[in_filename] = results_temp
infile.close()

halide_total = 0
nfm_total = 0

for filename in results:
	print filename
	temp = results[filename]
	halide = 0
	nfm = 0
	for name, (halide_counts, nfm_counts) in temp.iteritems():
		hcount = sum(halide_counts)
		nfmcount = sum(nfm_counts)
		halide += hcount
		nfm += nfmcount
		percentage = 0.0
		if (hcount != 0):
			percentage = ((nfmcount*100.0)/hcount)
		print '    Type: %-10s, Halide: %-10d, NFM: %-10d, NFM/halide: %.2f%%' % (name, hcount, nfmcount, percentage)

	halide_total += halide
	nfm_total += nfm
	print '  Halide: %d, NFM: %d, NFM/halide: %.2f%%' % (halide, nfm, ((nfm*100.0)/halide))

print ''
print 'Halide:', halide_total
print 'NFM:', nfm_total
print 'NFM/halide: %.2f%%' % ((nfm_total*100.0)/halide_total)