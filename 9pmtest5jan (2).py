from blob_detection import detect
from utils import triangulate, navigate, draw_pts,draw_pts2,draw_pts3,draw_pts4, draw_bot,draw_bot2,draw_bot3,draw_bot4, draw_rts,draw_rts2,draw_rts3,draw_rts4, display_msg, display_msg2,display_msg3,display_msg4,remove_vtm,draw_pts2, draw_bot2, draw_rts2,display_msg2,navigate2,navigate3,navigate4
import numpy as np
import cv2
import time
import requests
import datetime
from multiprocessing.dummy import Pool

pool = Pool(10) # Creates a pool with ten threads; more threads = more concurrency.
                # "pool" is a module attribute; you can be sure there will only
                # be one of them in your application
                # as modules are cached after initialization.

# api-endpoint
url1 = "http://192.168.137.48"#E094
url2 = "http://192.168.137.91"#e7d8



font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
# If the program is running in dev environment
dev_mode = True

print("Starting camera...")


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
frame_rate = 5

# Variable to store the previous time of analysed frame
prev = 0

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640,480))

# Load the mask for calibration and removing unwanted area from frames
mask = cv2.imread('images/mask2.png', 0)
calibrate = cv2.imread('images/calibrate.png', 0)



starting1 = True
restarting1 = False
starting11 = False
restarting11 = False
starting12 = False
restarting12 = False
starting13 = False
restarting13 = False
starting14 = False
restarting14 = False
starting15 = False
restarting15 = False
starting16 = False
restarting16 = False
starting17 = False
restarting17 = False
starting18 = False
restarting18 = False
starting19 = False
restarting19 = False
starting110 = False
restarting110 = False
starting111 = False
restarting111 = False
starting112 = False
restarting112 = False
starting113 = False
restarting113 = False
starting114 = False
restarting114 = False
starting115 = False
restarting115 = False
starting116 = False
restarting116 = False
starting117 = False
restarting117 = False
starting118 = False
restarting118 = False
starting119 = False
restarting119 = False
starting120 = False
restarting120 = False
restarting121 = False
starting121 = False
restarting122 = False
starting122 = False
restarting123 = False
starting123 = False
restarting124 = False
starting124 = False
restarting125 = False
starting125 = False
restarting126 = False
starting126 = False
starting127 = False
restarting127 = False
starting128 = False
restarting128 = False
starting129 = False
restarting129 = False
starting130 = False
restarting130 = False
finished1 = False

starting21 = True
restarting21 = False
starting211 = False
restarting211 = False
starting212 = False
restarting212 = False
starting213 = False
restarting213 = False
starting214 = False
restarting214 = False
starting215 = False
restarting215 = False
starting216 = False
restarting216 = False
starting217 = False
restarting217 = False
starting218 = False
restarting218 = False
starting219 = False
restarting219 = False
starting2110 = False
restarting2110 = False
starting2111 = False
restarting2111 = False
starting2112 = False
restarting2112 = False
starting2113 = False
restarting2113 = False
starting2114 = False
restarting2114 = False
starting2115 = False
restarting2115 = False
starting2116 = False
restarting2116 = False
starting2117 = False
restarting2117 = False
starting2118 = False
restarting2118 = False
starting2119 = False
restarting2119 = False
starting2120 = False
restarting2120 = False
restarting2121 = False
starting2121 = False
restarting2122 = False
starting2122 = False
restarting2123 = False
starting2123 = False
restarting2124 = False
starting2124 = False
restarting2125 = False
starting2125 = False
restarting2126 = False
starting2126 = False
starting2127 = False
restarting2127 = False
starting2128 = False
restarting2128 = False
starting2129 = False
restarting2129 = False
starting2130 = False
restarting2130 = False
finished2 = False






#robot 1

r1mumbai = [np.array([190,155]),np.array([190,116])]
r1mhome = [np.array([33,140])]

r1delhi = [np.array([410,155]),np.array([410,115])]
r1dhome = [np.array([410,155]),np.array([33,140])]

r1kolkata = [np.array([595,153]),np.array([596,110])]
r1khome = [np.array([580,153]),np.array([33,140])]

r1chennai = [np.array([200,150]),np.array([200,180])]
r1chome = [np.array([33,140])]

r1bengaluru = [np.array([400,155]),np.array([410,178])]
r1bhome = [np.array([410,155]),np.array([33,140])]

r1hyd = [np.array([530,153]),np.array([575,195])]
r1hhome = [np.array([530,153]),np.array([33,140])]


#robo 2
'''
r2pune = [np.array([200,370]),np.array([200,390])]
r2phome = [np.array([38,400])]

r2ahem = [np.array([410,370]),np.array([410,400])]
r2ahome = [np.array([410,370]),np.array([38,400])]

r2jaip = [np.array([530,370]),np.array([565,405])]
r2jhome = [np.array([530,370]),np.array([38,400])]


r2chennai = [np.array([200,370]),np.array([200,345])]
r2chome = [np.array([38,400])]

r2bengaluru = [np.array([410,370]),np.array([410,345])]
r2bhome = [np.array([410,370]),np.array([38,400])]

r2hyd = [np.array([580,383]),np.array([590,340])]
r2hhome = [np.array([580,383]),np.array([38,400])]
'''

r2ahem  = [np.array([405,370]),np.array([405,405])]
r2ahome = [np.array([410,370]),np.array([38,400])]

r2chennai = [np.array([200,370]),np.array([200,335])]
r2chome = [np.array([42,390])]

r2bengaluru = [np.array([405,370]),np.array([405,345])]
r2bhome =  [np.array([410,370]),np.array([42,390])]

r2pune = [np.array([180,350]),np.array([180,400])]
r2phome = [np.array([42,390])]

r2jaip = [np.array([530,370]),np.array([585,412])]
r2jhome = [np.array([530,370]),np.array([42,390])]

r2hyd = [np.array([580,383]),np.array([595,345])]
r2hhome = [np.array([575,383]),np.array([42,390])]


pos = np.array([60, 450])
dirn = np.array([0, -10])

pos2 = np.array([60, 450])
dirn2 = np.array([0, -10])
pos3 = np.array([60, 450])
dirn3 = np.array([0, -10])
pos4 = np.array([60, 450])
dirn4 = np.array([0, -10])

seq = 0
seq2 = 0
seq3 = 0
seq4 = 0

marker = []
marker2 = []

r1=0
r2=0
r3=0
r4=0
present = datetime.datetime.now()
# Main loop
while cap.isOpened():
	time_elapsed = time.time() - prev
	ret, frame = cap.read()
	dt = datetime.datetime.now()
	timer=dt-present
	fame=cv2.putText(frame, str(timer),(10, 30),font, 1,(210, 155, 155),2, cv2.LINE_8)
	
	

	if time_elapsed > 1./frame_rate and not finished1:
                
		prev = time.time()
		#Threshold of BLUE in HSV space 

		lower_blue = np.array([10, 120, 135])
		upper_blue = np.array([20,255,255])

		pts1 = detect(frame, mask, lower_blue, upper_blue)
		
		
		# Determine whether the robot is found
		if len(pts1) >= 3:
			pos, dirn = triangulate(pts1)

		

		if starting1:
			marker=[]
			marker = r1mumbai
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1mumbai[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1mumbai):
					seq = 0
					dist = 0
					starting1 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting1 = True


		elif restarting1:
			marker=[]
			marker = r1mhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1mhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1mhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting1 = False
					print("Robot-1 Reached source")
					starting11=True

		elif starting11:
			marker=[]
			marker = r1chennai
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1chennai[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1chennai):
					seq = 0
					dist = 0
					starting11 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached chennai")
					restarting11 = True


		elif restarting11:
			marker=[]
			marker = r1chome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1chome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1chome):
					seq = 0
					dist = 0
					restarting11= False
					r = requests.get(url1 + "/grab")
					
					print("Robot-1 Reached source")
					starting12=True

					
		elif starting12:
			marker=[]
			marker = r1bengaluru
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1bengaluru[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1bengaluru):
					seq = 0
					dist = 0
					starting12 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting12 = True

		elif restarting12:
			marker=[]
			marker = r1bhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1bhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1bhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting12 = False
					print("Robot-1 Reached source")
					starting13=True


		

		elif starting13:
			marker=[]
			marker = r1delhi
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1delhi[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1delhi):
					seq = 0
					dist = 0
					starting13 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached chennai")
					restarting13 = True


		elif restarting13:
			marker=[]
			marker = r1dhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1dhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1dhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting13 = False
					print("Robot-1 Reached source")
					starting14=True
					

		elif starting14:
			marker=[]
			marker = r1kolkata
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1kolkata[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1kolkata):
					seq = 0
					dist = 0
					starting14 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting14 = True


		elif restarting14:
			marker=[]
			marker = r1khome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1khome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1khome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting14 = False
					print("Robot-1 Reached source")
					starting15=True
					
		elif starting15:
			marker=[]
			marker = r1hyd
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1hyd[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1hyd):
					seq = 0
					dist = 0
					starting15 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached Chennai")
					restarting15 = True

		
		elif restarting15:
			marker=[]
			marker = r1hhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1hhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1hhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting15 = False
					print("Robot-1 Reached source")
					starting16 = True

		elif starting16:
			marker=[]
			marker = r1delhi
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1delhi[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1delhi):
					seq = 0
					dist = 0
					starting16 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached Chennai")
					restarting16 = True

		
		elif restarting16:
			marker=[]
			marker = r1dhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1dhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1dhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting16 = False
					print("Robot-1 Reached source")
					starting17 = True
		elif starting17:
			marker=[]
			marker = r1bengaluru
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1bengaluru[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1bengaluru):
					seq = 0
					dist = 0
					starting17 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached Chennai")
					restarting17 = True

		
		elif restarting17:
			marker=[]
			marker = r1bhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1bhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1bhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting17 = False
					print("Robot-1 Reached source")
					starting18 = True
		elif starting18:
			marker=[]
			marker = r1kolkata
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1kolkata[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1kolkata):
					seq = 0
					dist = 0
					starting18 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached Chennai")
					restarting18 = True

		
		elif restarting18:
			marker=[]
			marker = r1khome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1khome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1khome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting18 = False
					print("Robot-1 Reached source")
					starting19 = True
		elif starting19:
			marker=[]
			marker = r1delhi
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1delhi[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1delhi):
					seq = 0
					dist = 0
					starting19 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached Chennai")
					restarting19 = True

		
		elif restarting19:
			marker=[]
			marker = r1dhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1dhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1dhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting19 = False
					print("Robot-1 Reached source")
					starting110 = True


		elif starting110:
			marker=[]
			marker = r1mumbai
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1mumbai[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1mumbai):
					seq = 0
					dist = 0
					starting110= False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting110 = True

		
		elif restarting110:
			marker=[]
			marker = r1mhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1mhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1mhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting110 = False
					print("Robot-1 Reached source")
					starting111 = True
		
					
		elif starting111:
			marker=[]
			marker = r1chennai
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1chennai[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1chennai):
					seq = 0
					dist = 0
					starting111= False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting111 = True

		
		elif restarting111:
			marker=[]
			marker = r1chome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1chome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1chome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting111 = False
					print("Robot-1 Reached source")
					starting112 = True
		
                
		if starting112:
			marker=[]
			marker = r1mumbai
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1mumbai[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1mumbai):
					seq = 0
					dist = 0
					starting112 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting112 = True


		elif restarting112:
			marker=[]
			marker = r1mhome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1mhome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1mhome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting112 = False
					print("Robot-1 Reached source")
					starting113=True
		elif starting113:
			marker=[]
			marker = r1kolkata
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1kolkata[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1kolkata):
					seq = 0
					dist = 0
					starting113 = False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting114 = True


		elif restarting114:
			marker=[]
			marker = r1khome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1khome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1khome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting114 = False
					print("Robot-1 Reached source")
					starting115=True
		elif starting115:
			marker=[]
			marker = r1chennai
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1chennai[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1chennai):
					seq = 0
					dist = 0
					starting115= False
					r = requests.get(url1 + "/grab")
					print("Robot-1 Reached mumbai")
					restarting115 = True

		
		elif restarting115:
			marker=[]
			marker = r1chome
			#print(marker)
			rts, angle, dist = navigate(pos, dirn, r1chome[seq:seq+1])
			if dist < 20 or cv2.waitKey(1) == ord(' '):
				seq += 1
				if seq == len(r1chome):
					seq = 0
					dist = 0
					r = requests.get(url1 + "/grab")
					restarting115 = False
					print("Robot-1 Reached source")
					finished1 = True
				

		# Defining a params dict for the parameters to be sent to the API 
		parameters = {
			"angle": angle,
			"distance": dist
		} 
  
		# Sending get request and saving the response as response object 
		pool.apply_async(requests.get, [url1 + "/move", parameters])

		
	if not finished1:
		draw_pts(frame, marker)
		draw_bot(frame, pos, dirn)
		draw_rts(frame, pos, rts, angle, dist)

		display_msg(frame, "Angle={:.2f}, Distance={}".format(angle, dist), (0, 255, 0))


	if time_elapsed > 1./frame_rate and not finished2:               
		prev = time.time()
		#Threshold of RED in HSV space
		lower_magenta = np.array([45,70, 45])
		upper_magenta = np.array([86,255,255])
		pts2 = detect(frame, mask, lower_magenta, upper_magenta)
		
		
		if len(pts2) >= 3:
			pos2, dirn2 = triangulate(pts2)

		
		
				
		if starting21:
			marker2=[]
			marker2 = r2jaip
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2jaip[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2jaip):
					seq2 = 0
					dist2 = 0
					starting21 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 pune")
					restarting21 = True

		# Initiate the restarting sequence
		elif restarting21:
			marker2=[]
			marker2 = r2jhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2jhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2jhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting21 = False
					starting211=True
					
					print("Robot-2 Reached Source")

		elif starting211:
			marker2=[]
			marker2 = r2hyd
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2hyd[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2hyd):
					seq2 = 0
					dist2 = 0
					starting211 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 chennai")
					restarting211 = True

		# Initiate the restarting sequence
		elif restarting211:
			marker2=[]
			marker2 = r2hhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2hhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2hhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting211 = False
					starting212=True
					
					print("Robot-2 Reached Source")
		elif starting212:
			marker2=[]
			marker2 = r2bengaluru
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2bengaluru[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2bengaluru):
					seq2 = 0
					dist2 = 0
					starting212 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 pune")
					restarting212 = True

		# Initiate the restarting sequence
		elif restarting212:
			marker2=[]
			marker2 = r2bhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2bhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2bhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting212 = False
					starting213=True
					
					print("Robot-2 Reached Source")

					


		if starting213:
			marker2=[]
			marker2 = r2ahem
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahem[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahem):
					seq2 = 0
					dist2 = 0
					starting213 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting213 = True

		# Initiate the restarting sequence
		elif restarting213:
			marker2=[]
			marker2 = r2ahome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting213= False
					starting214=True
					
					print("Robot-2 Reached Source")

		if starting214:
			marker2=[]
			marker2 = r2bengaluru
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2bengaluru[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2bengaluru):
					seq2 = 0
					dist2 = 0
					starting214 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached pune")
					restarting214 = True

		# Initiate the restarting sequence
		elif restarting214:
			marker2=[]
			marker2 = r2bhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2bhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2bhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting214= False
					starting215=True
					
					print("Robot-2 Reached Source")
		if starting215:
			marker2=[]
			marker2 = r2chennai
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2chennai[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2chennai):
					seq2 = 0
					dist2 = 0
					starting215 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting215 = True

		# Initiate the restarting sequence
		elif restarting215:
			marker2=[]
			marker2 = r2chome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2chome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2chome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting215= False
					starting216=True
					
					print("Robot-2 Reached Source")
		if starting216:
			marker2=[]
			marker2 = r2pune
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2pune[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2pune):
					seq2 = 0
					dist2 = 0
					starting216 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting216 = True

		# Initiate the restarting sequence
		elif restarting216:
			marker2=[]
			marker2 = r2phome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2phome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2phome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting216= False
					starting217=True
					
					print("Robot-2 Reached Source")


		elif starting217:
			marker2=[]
			marker2 = r2ahem
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahem[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahem):
					seq2 = 0
					dist2 = 0
					starting217 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached pune")
					restarting217 = True

		# Initiate the restarting sequence
		elif restarting217:
			marker2=[]
			marker2 = r2ahome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting217 = False
					starting218=True
					
					print("Robot-2 Reached Source")

		
		elif starting218:
			marker2=[]
			marker2 = r2jaip
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2jaip[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2jaip):
					seq2 = 0
					dist2 = 0
					starting218 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached pune")
					restarting218 = True

		# Initiate the restarting sequence
		elif restarting218:
			marker2=[]
			marker2 = r2jhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2jhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2jhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting218 = False
					starting219=True
					
					print("Robot-2 Reached Source")

		elif starting219:
			marker2=[]
			marker2 = r2ahem
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahem[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahem):
					seq2 = 0
					dist2 = 0
					starting219 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached pune")
					restarting219= True

		# Initiate the restarting sequence
		elif restarting219:
			marker2=[]
			marker2 = r2ahome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting219 = False
					starting2110=True
					
					print("Robot-2 Reached Source")

		elif starting2110:
			marker2=[]
			marker2 = r2pune
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2pune[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2pune):
					seq2 = 0
					dist2 = 0
					starting2110= False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting2110= True

		# Initiate the restarting sequence
		elif restarting2110:
			marker2=[]
			marker2 = r2phome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2phome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2phome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting2110 = False
					starting2111=True
					
					print("Robot-2 Reached Source")
		elif starting2111:
			marker2=[]
			marker2 = r2hyd
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2hyd[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2hyd):
					seq2 = 0
					dist2 = 0
					starting2111= False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting2111= True

		# Initiate the restarting sequence
		elif restarting2111:
			marker2=[]
			marker2 = r2hhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2hhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2hhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting2111 = False
					starting2112=True
					
					print("Robot-2 Reached Source")
		elif starting2112:
			marker2=[]
			marker2 = r2bengaluru
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2bengaluru[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2bengaluru):
					seq2 = 0
					dist2 = 0
					starting2112= False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting2112= True

		# Initiate the restarting sequence
		elif restarting2112:
			marker2=[]
			marker2 = r2bhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2bhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2bhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting2112 = False
					starting2113=True
					
					print("Robot-2 Reached Source")
					

		elif starting2113:
			marker2=[]
			marker2 = r2jaip
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2jaip[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2jaip):
					seq2 = 0
					dist2 = 0
					starting2113 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 pune")
					restarting2113 = True

		# Initiate the restarting sequence
		elif restarting2113:
			marker2=[]
			marker2 = r2jhome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2jhome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2jhome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting2113 = False
					starting2114=True
					
					print("Robot-2 Reached Source")
		elif starting2114:
			marker2=[]
			marker2 = r2pune
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2pune[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2pune):
					seq2 = 0
					dist2 = 0
					starting2114 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached chennai")
					restarting2114 = True

		# Initiate the restarting sequence
		elif restarting2114:
			marker2=[]
			marker2 = r2phome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2phome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2phome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting2114= False
					starting2115=True
					
					print("Robot-2 Reached Source")


		elif starting2115:
			marker2=[]
			marker2 = r2ahem
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahem[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahem):
					seq2 = 0
					dist2 = 0
					starting2115 = False
					r = requests.get(url2 + "/grab")
					print("Robot-2 Reached pune")
					restarting2115 = True

		# Initiate the restarting sequence
		elif restarting2115:
			marker2=[]
			marker2 = r2ahome
			#print("-----------------------------------------------------")
			#print(marker2)
			rts2, angle2, dist2 = navigate2(pos2, dirn2, r2ahome[seq2:seq2+1])
			if dist2 < 20 or cv2.waitKey(1) == ord(' '):
				seq2 += 1
				if seq2 == len(r2ahome):
					seq2 = 0
					dist2 = 0
					r = requests.get(url2 + "/grab")
					restarting2115 = False
					finished2=True
					
					print("Robot-2 Reached Source")

		# Defining a params dict for the parameters to be sent to the API 
		parameters = {
			"angle": angle2,
			"distance": dist2
		} 
  
 
		pool.apply_async(requests.get, [url2 + "/move", parameters])

	
	if not finished2:
		draw_pts2(frame, marker2)
		draw_bot2(frame, pos2, dirn2)
		draw_rts2(frame, pos2, rts2, angle2, dist2)
		display_msg2(frame, "Angle={:.2f}, Distance={}".format(angle2, dist2), (255, 0, 0))

	out.write(frame)	
	cv2.imshow('frame', frame)
	c = cv2.waitKey(1)
	if c == ord('\x1b'):
		break
	

	

	

cap.release()


cv2.destroyAllWindows()
