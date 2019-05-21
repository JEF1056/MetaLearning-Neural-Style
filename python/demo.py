#!/usr/bin/env python

import _pycaffe as caffe
from cv2 import *
import pdb
import numpy as np
import time
from argparse import ArgumentParser
import os
from tqdm import tqdm
import shutil
import subprocess

MODEL_LOC = "/content/drive/My Drive/IMP_STYLE/train_8.caffemodel"
MODEL_PROTO = "prototxt/8/"
CONTENT_LOC = "/content/drive/My Drive/IMP_STYLE/image2.jpg"
STYLE_LOC = "/content/drive/My Drive/IMP_STYLE/1_zToNBcKp1777KbT4WtvnkA.jpeg"
OUT_LOC = "/content/drive/My Drive/IMP_STYLE/out.jpg"

def pycaffe_hidden(im_label):
	prototxt_file = MODEL_PROTO+'hidden.prototxt'
	weights_file = MODEL_LOC
	if caffe.is_initialized() < 1:
		caffe.init(prototxt_file, weights_file)
		caffe.set_device(0)
	
	im_label = im_label.astype('float32')

	im_label[:,:,0] = im_label[:,:,0] - 104.008
	im_label[:,:,1] = im_label[:,:,1] - 116.669
	im_label[:,:,2] = im_label[:,:,2] - 122.675
	im_label_ = np.expand_dims(im_label,3)
	im_label = np.transpose(im_label_,(3,2,0,1)).copy()


	input_data = [im_label]
	score = caffe.forward(0,input_data)
	hidden_feat = score[0].squeeze()
	
	return hidden_feat
	
def pycaffe_param(hidden_feat):
	prototxt_file = MODEL_PROTO+'param.prototxt'
	weights_file = MODEL_LOC
	if caffe.is_initialized() < 2:
		caffe.init(prototxt_file, weights_file)
		caffe.set_device(0)
	
	hidden_feat = hidden_feat.reshape((1,hidden_feat.size,1,1))
	input_data = [hidden_feat]
	param = caffe.forward(1, input_data)
	
	
	caffe.save_model(param,'layer_name.txt','base.caffemodel','predict.caffemodel')

def pycaffe_predict(im):
	prototxt_file = MODEL_PROTO+'predict.prototxt'
	weights_file = 'predict.caffemodel'
	if caffe.is_initialized() < 3:
		caffe.init(prototxt_file, weights_file)
		caffe.set_device(0)
		
	im = im.astype('float32')
	im[:,:,0] = im[:,:,0] - 104.008
	im[:,:,1] = im[:,:,1] - 116.669
	im[:,:,2] = im[:,:,2] - 122.675
	im = np.expand_dims(im,3)
	im = np.transpose(im,(3,2,0,1)).copy()
	
	input_data = [im]
	#t1=time.time()
	score = caffe.forward(2, input_data)
	#t2=time.time()
	#print(t2-t1)
	
	raw_score = score[0]
	
	raw_score = raw_score[0,:,:,:]
	
	raw_score = np.transpose(raw_score,(1,2,0)).copy()
	
	raw_score[:,:,0] = raw_score[:,:,0] + 104.008
	raw_score[:,:,1] = raw_score[:,:,1] + 116.669
	raw_score[:,:,2] = raw_score[:,:,2] + 122.675
	
	raw_score = np.clip(raw_score,0,255)
	
	return raw_score.astype('uint8')
	
def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--model', type=str,
			    dest='model', help='dir to find the model',
			    metavar='MODEL_LOC', required=True)
	parser.add_argument('--prototxt', type=str,
			    dest='prototxt', help='dir to find the model',
			    metavar='MODEL_PROTO', required=True)
	parser.add_argument('--content', type=str,
			    dest='content', help='dir to find content image/video',
			    metavar='CONTENT_LOC')
	parser.add_argument('--style', type=str,
			    dest='style', help='dir to find style image',
			    metavar='STYLE_LOC', required=True)
	parser.add_argument('--out', type=str,
			    dest='out', help='dir to save output',
			    metavar='OUT_LOC')
	parser.add_argument('--oc', dest='oc', help='original colors',
			    action='store_true')
	parser.add_argument('--cct', type=str,
			    dest='cct', help='Convert color type, of options yuv, lab, luv, and ycrcb', 
			    default="yuv")
	parser.add_argument('--cr', type=float,
			    dest='cr', help='content ratio', 
			    default=1)
	parser.add_argument('--sr', type=float,
			    dest='sr', help='style ratio', 
			    default=1)			
	parser.add_argument('--video', dest='video', help='uwu for those video fans', 
			    action='store_true')
	parser.add_argument('--realtime', dest='realtime', help='UWU IS THAT REALTIME?!?!', 
			    action='store_true')
	parser.add_argument('--camera', type=int, dest='camera', help='OMG A CAMERA OWO')
	parser.set_defaults(gpu=False, video=False, oc=False, realtime=False, camera=0)
	return parser
	
	
if __name__ == '__main__':
	parser = build_parser()
	options = parser.parse_args()
	
	MODEL_LOC=options.model
	MODEL_PROTO=options.prototxt

	caffe.base_model(options.model, 'base.txt')

	style_im = imread(options.style)
	style_im = cv2.resize(style_im, (0,0), fx=options.sr, fy=options.sr)
	print("Style size: " + str(style_im.shape))
	hidden_feat = pycaffe_hidden(style_im)
	
	pycaffe_param(hidden_feat)
	
	if options.video == True and options.realtime == True:
		print("Cannot have both video and realtime active at the same time")
		end
	
	if options.video:
		try:
			shutil.rmtree("recon")	
		except:
			pass
		os.mkdir("recon")
		vidcap = cv2.VideoCapture(options.content)
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		if int(major_ver)  < 3 :
			fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
		else :
			fps = vidcap.get(cv2.CAP_PROP_FPS)
		success,image = vidcap.read()
		video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		print("Found " + str(video_length) + " frames")
		height, width, layers = image.shape
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		video = cv2.VideoWriter("recon/NOAUD.avi", fourcc, fps, (width,height))

		i = 0
		pbar = tqdm(total=video_length)
		while(vidcap.isOpened()) and success == True and i < video_length:
			if success==True:
				origin_im = cv2.resize(image, (0,0), fx=options.cr, fy=options.cr) 
				if i == 0:
					print("Content size: " + str(origin_im.shape))
				scoremap = pycaffe_predict(origin_im)
				if options.oc:
					if options.cct == 'yuv':
					  cvt_type = cv2.COLOR_BGR2YUV
					  inv_cvt_type = cv2.COLOR_YUV2BGR
					elif options.cct == 'ycrcb':
					  cvt_type = cv2.COLOR_BGR2YCR_CB
					  inv_cvt_type = cv2.COLOR_YCR_CB2BGR
					elif options.cct == 'luv':
					  cvt_type = cv2.COLOR_BGR2LUV
					  inv_cvt_type = cv2.COLOR_LUV2BGR
					elif options.cct == 'lab':
					  cvt_type = cv2.COLOR_BGR2LAB
					  inv_cvt_type = cv2.COLOR_LAB2BGR
					content_cvt = cv2.cvtColor(origin_im, cvt_type)
					stylized_cvt = cv2.cvtColor(scoremap, cvt_type)
					c1, _, _ = cv2.split(stylized_cvt)
					_, c2, c3 = cv2.split(content_cvt)
					merged = cv2.merge((c1, c2, c3))
					scoremap = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
				video.write(np.uint8(scoremap))
			pbar.update(1)
			success,image = vidcap.read()
			i += 1
			
		vidcap.release()
		pbar.close()
		cv2.destroyAllWindows()
		video.release()

		#Extract audio
		subprocess.call(['ffmpeg', '-i', options.content, '-f', 'mp3', '-ab', '192000', '-vn', 'recon/v_aud.mp3'])
		subprocess.call(['ffmpeg', '-i', "recon/NOAUD.avi", '-i', 'recon/v_aud.mp3', '-vcodec', 'x265', '-crf', '24', '-map', '0:0', '-map', '1:0', '-c:v', 'copy', '-c:a', 'copy', options.out])	
	if options.realtime:
		vidcap = cv2.VideoCapture(options.camera)
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		
		fpscount=0
		t1 = time.time()
		while(True):
			success,image = vidcap.read()
			origin_im = cv2.resize(image, (0,0), fx=options.cr, fy=options.cr) 
			if fpscount == 0:
				print("Content size: " + str(origin_im.shape))
			scoremap = pycaffe_predict(origin_im)
			if options.oc:
				if options.cct == 'yuv':
				  cvt_type = cv2.COLOR_BGR2YUV
				  inv_cvt_type = cv2.COLOR_YUV2BGR
				elif options.cct == 'ycrcb':
				  cvt_type = cv2.COLOR_BGR2YCR_CB
				  inv_cvt_type = cv2.COLOR_YCR_CB2BGR
				elif options.cct == 'luv':
				  cvt_type = cv2.COLOR_BGR2LUV
				  inv_cvt_type = cv2.COLOR_LUV2BGR
				elif options.cct == 'lab':
				  cvt_type = cv2.COLOR_BGR2LAB
				  inv_cvt_type = cv2.COLOR_LAB2BGR
				content_cvt = cv2.cvtColor(origin_im, cvt_type)
				stylized_cvt = cv2.cvtColor(scoremap, cvt_type)
				c1, _, _ = cv2.split(stylized_cvt)
				_, c2, c3 = cv2.split(content_cvt)
				merged = cv2.merge((c1, c2, c3))
				scoremap = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
			t2 = time.time()
			fpscount=fpscount+1
			fps = (fpscount/(t2-t1))
			font = cv2.FONT_HERSHEY_SIMPLEX
			withfps = cv2.putText(cv2.resize(np.uint8(scoremap), (0,0), fx=1.5, fy=1.5),str(round(fps,2))+" fps",(10,40), font, 1,(255,255,255),2,cv2.LINE_AA)
			cv2.imshow('frame', withfps)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		vidcap.release()
		cv2.destroyAllWindows()
	else:
		origin_im = cv2.resize(cv2.imread(options.content), (0,0), fx=options.cr, fy=options.cr) 
		if i == 0:
			print("Content size: " + str(origin_im.shape))
		t1=time.time()
		scoremap = pycaffe_predict(origin_im)
		if options.oc:
			if options.cct == 'yuv':
				cvt_type = cv2.COLOR_BGR2YUV
				inv_cvt_type = cv2.COLOR_YUV2BGR
			elif options.cct == 'ycrcb':
				cvt_type = cv2.COLOR_BGR2YCR_CB
				inv_cvt_type = cv2.COLOR_YCR_CB2BGR
			elif options.cct == 'luv':
				cvt_type = cv2.COLOR_BGR2LUV
				inv_cvt_type = cv2.COLOR_LUV2BGR
			elif options.cct == 'lab':
				cvt_type = cv2.COLOR_BGR2LAB
				inv_cvt_type = cv2.COLOR_LAB2BGR
			content_cvt = cv2.cvtColor(origin_im, cvt_type)
			stylized_cvt = cv2.cvtColor(scoremap, cvt_type)
			c1, _, _ = cv2.split(stylized_cvt)
			_, c2, c3 = cv2.split(content_cvt)
			merged = cv2.merge((c1, c2, c3))
			dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
			cv2.imwrite(options.out, dst)
		else:
			imwrite(options.out,scoremap)
		print("Took " + str(round((time.time()-t1),2)) + " seconds")
	print("DONE")
