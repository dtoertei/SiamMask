import os
import argparse


parser = argparse.ArgumentParser(description='Script for video processing with ffmpeg')

parser.add_argument('--split', default=False, action = 'store_true', help='split video into frames (True) or make gif (False)')
parser.add_argument('--split_dir', default='../../data/fashion_MNIST', type=str, help='path to test video from which frames will be extracted')
parser.add_argument('--split_name', default='fashion_show.mp4', type=str, help='test video name from which frames will be extracted')
parser.add_argument('--gif_frames_dir', default='../../data/fashion_MNIST/test_frames', type=str, help='path to the resulting frames after applying MNIST CNN')
parser.add_argument('--gif_dir', default='../../data/fashion_MNIST/GIFs', type=str, help='path to the resulting GIF file')
args = parser.parse_args()


# Demo video may be found here: https://www.youtube.com/watch?v=6npPggFsz8A
def ffmpeg_split(video_file_dir, video_file_name):

	video_file_path = os.path.join(video_file_dir, video_file_name)
	frames_dir = os.path.join(video_file_dir, 'frames')
	gifs_dir = os.path.join(video_file_dir, 'GIFs')

	if not os.path.exists(frames_dir):
		try:
			os.mkdir(frames_dir)
		except OSError:
			print("%s directory creation failed" % frames_dir)
		else:
			print("%s directory created successfully" % frames_dir)

	
	try:
		# optimize as png files are large
		os.system("ffmpeg -t 00:03:00 -ss 00:01:15 -i {0} -r \"10\" {1}".format(video_file_path, os.path.join(frames_dir, 'thumb%04d.png')))
	except OSError:
		print("Splitting of video with ffmpeg failed")
	else:
		print("Splitting of video with ffmpeg successful")



# Requires specifying of the starting frame
def makeGIF(frames_dir, gif_dir, start_frame):

	video_file_path = os.path.join(frames_dir, 'out.mp4')
	gif_file_path = os.path.join(gif_dir, 'out.gif')


	try:
		os.system("ffmpeg -y -start_number {0} -i {1} -c:v libx264 -pix_fmt yuv420p {2}".format(start_frame, os.path.join(frames_dir, 'thumb%04d.png'), video_file_path))
	except OSError:
		print("Merging of frames into video with ffmpeg failed")
	else:
		print("Merging of frames into video with ffmpeg successful")


	if not os.path.exists(gif_dir):
		try:
			os.mkdir(gif_dir)
		except OSError:
			print("%s directory creation failed" % gif_dir)
		else:
			print("%s directory created successfully" % gif_dir)


	try:
		os.system("ffmpeg -y -i {0} {1}".format(video_file_path, gif_file_path))
	except OSError:
		print("Video to gif conversion with ffmpeg failed")
	else:
		print("Video to gif conversion with ffmpeg success")

	# video is needed for a to-gif conversion. one may remove it but I find it useful to play with ffmpeg settings and get the right fps for the resulting GIF.
	# uncomment these lines to remove the video:
	#try:
	#	os.remove(video_file_path)
	#except OSError:
	#	print("Remove of intermediary video file failed")
	#else:
	#	print("Remove of intermediary video file success")


# from within $SiamMask/experiments/siammask_sharp
if __name__ == '__main__':

	
	if(args.split):
		ffmpeg_split(args.split_dir, args.split_name)
	else:
		print(args.split)
		dress_start_frame = 1526
		sandal_start_frame = 949
		top_start_frame = 99
		makeGIF(args.gif_frames_dir, args.gif_dir, dress_start_frame)



