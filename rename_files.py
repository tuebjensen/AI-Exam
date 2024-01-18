import os


i = 0
path="C:\\Users\\TJ\\Desktop\\Programming\\Examprep\\Exam\\data\\square\\"
for filename in os.listdir(path):
	my_dest ="Square_" + str(i) + ".png"
	my_source =path + filename
	my_dest =path + my_dest

	os.rename(my_source, my_dest)
	i += 1
