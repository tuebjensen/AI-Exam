import os
# Function to rename multiple files
def main():
	i = 0
	path="C:\\Users\\TJ\\Desktop\\Programming\\Examprep\\Exam\\data\\square\\"
	for filename in os.listdir(path):
		my_dest ="Square_" + str(i) + ".png"
		my_source =path + filename
		my_dest =path + my_dest
		# rename() function will
		# rename all the files
		os.rename(my_source, my_dest)
		i += 1
# Driver Code
if __name__ == '__main__':
	# Calling main() function
	main()