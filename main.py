#Segment extraction try building from trainingset-a first
import argparse
import random
import os
import librosa
import scipy
from python_speech_features import mfcc
import h5py
import numpy as np
from tqdm import tqdm


#load parser help guide
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/billyzhaoyh/Desktop/AI_for_cardio_arrest/SegmentExtraction/training', help="Directory with the ingredient dataset")
parser.add_argument('--statelabel_dir', default='/Users/billyzhaoyh/Desktop/AI_for_cardio_arrest/SegmentExtraction/segment', help="Directory with the matlab annotations of four different states")
parser.add_argument('--output_dir', default='/Users/billyzhaoyh/Desktop/AI_for_cardio_arrest/mfcc_training', help="Where to write the new data")

def extract_mfcc(wav_path,mat_path,cycle=5):
	mat=scipy.io.loadmat(mat_path)
	audio_segment=mat['state_ans']
	y, sr = librosa.load(wav_path, sr=44100)
	y_2k = librosa.resample(y, sr, 2000)
	#locate the start of a heart sound - how many 5 cycles does this sample contain from start to finish, from first S1 to last Diastole
	#locate the first S1: strip out the label column and find the index for the first S1
	labels=audio_segment[:,1]
	indexes=audio_segment[:,0]-1 #python matlab mismatch
	#can trust the system lable sequencing is correct (no mechanism of missing a label in HMM)
	#find all the s1 locations in the labels and their corresponding indexes and store their corresponding indexed in data points in groups
	#only keep the label values
	n=len(labels)
	label_value=[]
	seg_index=[]
	for i in range(n):
		individual_label=labels[i][0]
		label_value.append(individual_label)
		individual_index=np.asscalar(indexes[i])
		seg_index.append(individual_index)
	groups=[] #keep track of the arrays of groups here
	indices_s1 = [i for i, x in enumerate(label_value) if x == "S1"]
	#check if a list is empty if it is empty then break out of the function
	mfcc_coeff_collage=[]
	if not indices_s1:
		return mfcc_coeff_collage

	#check if last indices is out of range to avoid running into error in the for loop below
	else:

		if (indices_s1[-1]+5)>len(label_value):
			del indices_s1[-1]

		#now we are ready to fill the groups
		for index in indices_s1:
			subgroup=[seg_index[index],seg_index[index+1],seg_index[index+2],seg_index[index+3],seg_index[index+4]]
			groups.append(subgroup)

		number_of_samples=len(groups)//cycle #number of samples we can extract from this wav file


		for i in range(0,number_of_samples,cycle):
			#here we parse out all the samples from this dataset and store them in the output directory.
			cycle_to_viz=groups[i]+groups[i+1]+groups[i+2]+groups[i+3]+groups[i+4]
			t_start=cycle_to_viz[0]
			t_end=cycle_to_viz[-1]
			segment_cycle=y_2k[t_start:t_end]
			mfcc_coeff = mfcc(segment_cycle,2000,winlen=0.025, winstep=0.01)
			mfcc_coeff_collage.append(mfcc_coeff)
		return mfcc_coeff_collage
		
if __name__ == '__main__':
	#initilaise parser help guid
	args = parser.parse_args()

	#check if dataset directory is valid
	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
	#name of all the folders that contain data 
	data_set = ['training-a', 'training-b', 'training-c', 'training-d','training-e','training-f']
	for folder in tqdm(data_set):
		raw_data_dir = os.path.join(args.data_dir, folder)
		mat_folder_dir=os.path.join(args.statelabel_dir, folder+'_StateAns')
		# Get the records file in RawData repositry
		normal_record=os.path.join(raw_data_dir, 'RECORDS-normal')
		abnormal_record=os.path.join(raw_data_dir, 'RECORDS-abnormal')
		with open(normal_record) as fp: normal = fp.read().splitlines()
		with open(abnormal_record) as fp: abnormal = fp.read().splitlines()
		mfcc_normal=[]
		mfcc_abnormal=[]
		for file in tqdm(normal):
			#go through each file 
			file_path = os.path.join(raw_data_dir,file+'.wav')
			mat_path=os.path.join(mat_folder_dir,file+'_StateAns.mat')
			mfcc_collage=extract_mfcc(file_path,mat_path)
			mfcc_normal=mfcc_normal+mfcc_collage
		for file in tqdm(abnormal):
			file_path = os.path.join(raw_data_dir,file+'.wav')
			mat_path=os.path.join(mat_folder_dir,file+'_StateAns.mat')
			mfcc_collage=extract_mfcc(file_path,mat_path)
			mfcc_abnormal=mfcc_abnormal+mfcc_collage
		#now we have all the mfcc coefficients for both sets of data and now we store all the data in output folder
		with h5py.File(os.path.join(args.output_dir, 'data_all.h5')) as hf:
			folder_group = hf.create_group(folder)
			i=1
			for mfcc_ind in mfcc_normal:
				name='normal'+str(i)
				folder_group.create_dataset(name,data=mfcc_ind)
				i=i+1
			i=1
			for mfcc_ind in mfcc_abnormal:
				name='abnormal'+str(i)
				folder_group.create_dataset(name,data=mfcc_ind)
				i=i+1
