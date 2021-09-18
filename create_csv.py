import csv
import os
from imutils import paths
import random

# list all the images in folder ./dataset
def create_csv(path, cls, train_w, test_w):
	# img_train = './data/unmask'
	img_train = path

	# cls = 0 #(mask, unmask, none, incorrect)
	img_name_train = list(paths.list_images(img_train))
	print(img_name_train[0])
	random.shuffle(img_name_train)
	# print(len(img_name_train))

	# divide the dataset into 2 part: training and testing
	# Number of training images/ testing iamges ~ 7/3
	num_img = len(img_name_train)
	num_train = int(num_img*90/100)
	# num_train = int(num_img)
	print('num_img', num_img)
	print('num_train', num_train)
	print('num_test', num_img - num_train)

	# train = open('unmask.csv', 'a')
	# train_w = csv.writer(train)
	# test = open('unmask_test.csv', 'a')
	# test_w = csv.writer(test)
	for i in range(0, num_img):
	  img_path = img_name_train[i]
	  if i < num_train: 
	    train_w.writerow([img_path, cls])
	  else:
	    test_w.writerow([img_path, cls])

def main():
	root = 'data'
	folders = os.listdir(root)

	train = open('train.csv', 'a')
	train_w = csv.writer(train)
	test = open('test.csv', 'a')
	test_w = csv.writer(test)
	label = open('label.csv', 'a')
	label_w = csv.writer(label)

	for idx, fd in enumerate(folders):
		path = f'{root}/{fd}'
		create_csv(path, idx, train_w, test_w)
		label_w.writerow([fd, idx])

main()
