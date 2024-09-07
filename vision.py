#Autor: Simon Eich
#File with the logic of the actual visin System. 

import cv2
import numpy as np
from math import sqrt
import time

# Import the global variable from the globals module
global indices
from globals import chip_quality_array, indices, camera_index
camera_index=0
vision_data = []


#indices = []
passed=0
missed=[]
jumped= 0


def get_vision_data_func(indices):
        
        filename="txt/config.txt"
        #image_path = f'Studie/img/img30.jpg' #used for tests when the camera is not available 

        #Patameters for the line, mask center movement and Changing of the Radius
        line_thickness = 4
        center_move_x = 7
        center_move_y= 3
        var_radius= 2

        # Parameters for square detection
        min_square_size = 2000  # Minimum area of the square
        max_square_size = 4000  # Maximum area of the square


        def capture_picture(camera_index):
            # Initialize the camera
            cap = cv2.VideoCapture(camera_index)

            # Check if the camera opened successfully
            if not cap.isOpened():
                raise Exception(f"Could not open camera with index {camera_index}")

            # Capture a single frame
            ret, frame = cap.read()

            # Release the camera
            cap.release()

            # Check if the frame was captured successfully
            if not ret:
                raise Exception("Failed to capture image")

            return frame


        def find_circle(image):
            
            
            # Read the image
            image = frame
            #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            


            # Apply a Gaussian blur to the image to reduce noise and improve circle detection
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Use the Hough Circle Transform to detect circles
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                       param1=50, param2=30, minRadius=150, maxRadius=350)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    center = (x, y)
                    radius = r
                    break  # Consider only the first detected circle
                
                # Create a mask to isolate the circle
                mask = np.zeros_like(gray)
                cv2.circle(mask, center, radius +2 , 255, -1)  # Draw a filled white circle on the mask

                # Extract the circular region from the image
                circular_region = cv2.bitwise_and(image, image, mask=mask)
                # Convert the circular region to grayscale
                circular_region_gray = cv2.cvtColor(circular_region, cv2.COLOR_BGR2GRAY)

                # Convert the modified circular region to grayscale
                circular_region_mono = cv2.cvtColor(circular_region, cv2.COLOR_BGR2GRAY)
                circular_region_mono_colored = cv2.cvtColor(circular_region_mono, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

                # Create an inverted mask to make the outside white
                inverted_mask = cv2.bitwise_not(mask)
                # Create a white background image
                white_background = np.full_like(image, 255)

                # Combine the monochrome circular region with the white background using the mask
                circle_image = cv2.bitwise_and(circular_region_mono_colored, circular_region_mono_colored, mask=mask) + \
                               cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

            else: 
                print("Error: No circles detected.")
                
            
            return circle_image, center, radius, image

        def black_image(image):
            # Create a black background image
            black_image = image.copy()
            height, width, channels = black_image.shape
            black_image = np.zeros((height, width, channels), dtype=np.uint8)

            return black_image

        def line_image_preparation(circle_image):
            # Prepare the image that a successful line detection is possible.
            image_line = circle_image.copy()
            #image_line = np.zeros_like(circle_image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

         # Use the Canny edge detector
            edges = cv2.Canny(image_line, 50, 150, apertureSize=3)
            # Dilate
            dilated_edges = cv2.dilate(edges, None, iterations=1) 

            #finde alle äusseren Konturen und markiere diese im Bild
            cnts, hir = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dilate 
            dilated_edges = cv2.dilate(edges, None, iterations=1)
            cv2.drawContours(dilated_edges, cnts, -1, (0, 0, 0), 2)   
            blur = cv2.GaussianBlur(dilated_edges, (21, 21), 2)    
            line_search_image = cv2.dilate(blur, None, iterations=1) 
            threshold_value = 128  # Modify if you want a different threshold
            _, line_search_image = cv2.threshold(line_search_image, threshold_value, 255, cv2.THRESH_BINARY)

            return line_search_image, dilated_edges

        def find_lines(line_search_image):

            lines = cv2.HoughLines(line_search_image, rho=1, theta=np.pi/10, threshold=220)
            # Draw detected lines
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)
                    cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)
            black_image_white_lines=black_image.copy()
            
            return black_image_white_lines, lines

        def find_last_line(white_black_lines, dilated_edges, black_image):

            black_image_background = image.copy()
            height, width, channels = black_image_background.shape
            black_image_background = np.zeros((height, width, channels), dtype=np.uint8)


            def filter_lines_by_length_and_orientation(lines, max_length, vertical_threshold):
                """Filter lines by maximum length and ensure they are approximately vertical."""
                filtered_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    # Check if the line is vertical
                    if length <= max_length and abs(x2 - x1) <= vertical_threshold:
                        filtered_lines.append(line)
                return filtered_lines

            def find_leftmost_line(lines):
                """Find the line with the smallest x-coordinate."""
                leftmost_line = None
                min_x = float('inf')

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate the smallest x-coordinate of the line
                    line_min_x = min(x1, x2)
                    if line_min_x < min_x:
                        min_x = line_min_x
                        leftmost_line = line
                return leftmost_line


            

            # Parameters for HoughLinesP
            threshold = 1        # Minimum number of intersections to detect a line
            minLineLength = 600     # Minimum length of a line segment
            maxLineGap = 10        # Maximum allowed gap between points on the same line
            maxLineLength = 1000    # Maximum length of line to be considered
            vertical_threshold = 1  # Threshold for vertical lines (difference in x-coordinates)

            dilated_edges = cv2.dilate(dilated_edges, None, iterations=2) 
            # Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(dilated_edges, 10, np.pi / 10, threshold, minLineLength, maxLineGap)
            # Filter lines by length and orientation
            filtered_lines = filter_lines_by_length_and_orientation(lines, maxLineLength, vertical_threshold) if lines is not None else []

            # Find the leftmost line
            leftmost_line = find_leftmost_line(filtered_lines)
            

            # Draw the leftmost line
            if leftmost_line is not None:
                x1, y1, x2, y2 = leftmost_line[0]
                cv2.line(black_image_background, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)

            white_black_lines=black_image 
            black_image_background = cv2.cvtColor(black_image_background, cv2.COLOR_BGR2GRAY)

            lines = cv2.HoughLines(black_image_background, rho=1, theta=np.pi/10, threshold=19)
            # Draw detected lines
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), line_thickness+20)


            return white_black_lines

        def circle_lines_SW(white_black_lines):
            new_center = (center[0]-center_move_x, center[1]-center_move_y)
            gray = cv2.cvtColor(white_black_lines, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            cv2.circle(mask, new_center, radius -var_radius , 255, -1)  # Draw a filled white circle on the mask
            circular_region = cv2.bitwise_and(white_black_lines, white_black_lines, mask=mask)

            return circular_region
        
        def change_background_to_white(circular_region):
            # Convert the image to grayscale
            gray = cv2.cvtColor(circular_region, cv2.COLOR_BGR2GRAY)

            # Apply a binary threshold to get a binary image
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Assume the largest contour by area is the outer boundary
                largest_contour = max(contours, key=cv2.contourArea)

                # Create a mask for the largest contour
                mask = np.zeros_like(white_black_lines, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                # Change the background to white
                white_background = np.ones_like(image, dtype=np.uint8) * 255
                square_image = np.where(mask == (255, 255, 255), circular_region, white_background)
            else:
                print("No contours found in the image.")
            

            return square_image


        def is_square(contour, epsilon=0.04):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) == 4:
                return True, approx
            return False, None

        def detect_squares(square_image, min_square_size, max_square_size, indices):
        
            gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edged = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            squares = []
            centers_squares=[]
            angle_array=[]

            for contour in contours:
                is_square_contour, approx = is_square(contour)
                if is_square_contour:
                    area = cv2.contourArea(contour)
                    if min_square_size <= area <= max_square_size:
                        # Get the bounding box
                        (x, y, w, h) = cv2.boundingRect(approx)
                        # Find the center
                        cx = int(x + w / 2)
                        cy = int(y + h / 2)
                        
                        # Calculate the angle
                        rect = cv2.minAreaRect(contour)
                        angle_point = 0 #rect[2]-90
                        angle_array.append(angle_point)          
                        
                        
                        squares.append((approx, (cx, cy), (x, y, w, h)))
                        centers_squares.append((cx,cy))
        
            num_array = np.array(angle_array)
            angle = np.mean(num_array)
            angle= float(angle)
            # Sort squares by their top-left corner positions (reading order: top to bottom, left to right)
            squares.sort(key=lambda s: (s[2][0], -s[2][1]))


            # Draw the squares, their centers, and numbers
            for idx, (square, center, bbox) in enumerate(squares, start=1):
                 x, y, w, h = bbox
                 cv2.drawContours(image, [square], -1, (0, 0, 255), 2)
                 cv2.circle(image, center, 5, (0, 0, 0), -1)

                 positions = [index for index, value in enumerate(indices) if value == '1']
                 increased_positions = [x + 1 for x in indices]
                 print(f'indicbbbies{increased_positions}')


                 # Check if idx is in indices and set the color
                 if idx in increased_positions:
                     color = (0, 255, 0)  # Red color for indices in the list
                 else:
                     print(idx)
                     color = (255, 0, 0)  # Yellow color for indices not in the list

                 cv2.putText(image, str(idx), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Draw the number
                # if idx == 2:
                #     cv2.putText(image, str(idx), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                # else:
                #     cv2.putText(image, str(idx), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            detect_squares_img=image.copy()
            
            num_center= len(centers_squares)
            #detect_squares_img = cv2.rotate(detect_squares_img, cv2.ROTATE_90_CLOCKWISE)
            

            #cv2.imshow("Detected Squares", detect_squares_img)


            return centers_squares, center, num_center, angle, detect_squares_img


        def read_config_file(filename):
        # getting the Matrix from the config file to do the coordinate transformation.
            points = []
            matrix = []
            transformed_p4 = None

            with open(filename, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    # Extract points (P1, P2, P3, P4)
                    if line.startswith('P'):
                        parts = line.split(':')
                        point = tuple(map(int, parts[1].strip().split(',')))
                        points.append(point)

                    # Extract the transformation matrix rows
                    elif line.strip() and not line.startswith('Transformation Matrix') and not line.startswith('Transformed P4') and not line.startswith('Time of configuration'):
                        row = list(map(float, line.strip().split()))
                        matrix.append(row)

                    # Extract transformed P4
                    elif line.startswith('Transformed P4'):
                        parts = line.split(':')
                        transformed_p4 = tuple(map(float, parts[1].strip().split(',')))

            # Convert matrix to a NumPy array
            matrix = np.array(matrix, dtype=np.float32)

            return matrix


        def center_transformation(centers_squares, center, angle):
            vision_data_center = []

            def sort_tuples(arr):
                return sorted(arr, key=lambda x: (x[0], x[0]))

            # Sorting the arrays
            sorted_center_squares = sort_tuples(centers_squares)

            for i, (center) in enumerate(sorted_center_squares):
                point= sorted_center_squares[i]
                vision_data_center = apply_transformation(matrix, point, vision_data_center, angle)

            return vision_data_center
            
        def apply_transformation(matrix, point, vision_data_center, angle):
            pts = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
            transformed_pts = cv2.transform(pts, matrix)
            
            # Extracting the numbers from the list
            numbers = transformed_pts[0][0]

            # Converting the np.float32 numbers to regular Python floats and rounding them
            rounded_values = (round(float(numbers[0]), 1), round(float(numbers[1]), 1))
            x=round(float(numbers[0]), 1)
            y=round(float(numbers[1]), 1)
            w=angle
            
            vision_data_center.append((x,y,w))
            
            return vision_data_center

            
        def filter_points_by_quality(coords, qual, center):
            # Check if both arrays have the same length
            if len(coords) != len(qual):

                # Filter points where quality is 1
                filtered_points = coords[qual == 1]
                print(filtered_points)

                for idx, (square, filtered_points, bbox) in enumerate(center, start=1):
                    x, y, w, h = bbox
                    cv2.drawContours(image, [square], -1, (0, 255, 0), 2)
                    cv2.circle(image, center, 5, (0, 0, 255), -1)
                    # Draw the number
                    cv2.putText(image, str(idx), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            return filtered_points
 




        # Analyze the provided image
        frame = capture_picture(camera_index)
        circle_image, center, radius, image= find_circle(frame)
        black_image = black_image(image)
        search_line_image, dilated_edges = line_image_preparation(circle_image ) 
        line_image = find_lines(search_line_image)
        white_black_lines = find_last_line(line_image, dilated_edges, black_image)
        circular_region= circle_lines_SW(white_black_lines)
        square_image = change_background_to_white(circular_region)
        detect_squares_centers, center, num_center, angle, detect_squares_img = detect_squares(square_image, min_square_size, max_square_size, indices)
        matrix = read_config_file(filename)
        #filtered_points = filter_points_by_quality(center, chip_quality_array, center) #The points are filtered later in the Karol Programm of the Robot. So it's not nessecary to do it here.
        
        
        if (num_center==36):
            vision_data = center_transformation(detect_squares_centers, center, angle)
            return vision_data, detect_squares_img, num_center
        else: 
            print(f'The number of detected Chips is {num_center}')
            vision_data = center_transformation(detect_squares_centers, center, angle)
            
            
            
            return vision_data, detect_squares_img, num_center

indices=[]

def get_vision_data(indices):

    indices=indices
    print
    num_center =0
    i=0
    #count_good=0
    #count_bad=0
    
    #The vision system doesn't have a succsrate dependig on the light about above 85%, 
    #so we need to run the vision system multiple times to be sure to get the result in fast time.
    while num_center!=36 and i<20:
        i+=1
        vision_data, detected_img, num_center = get_vision_data_func(indices)
        time.sleep(1)  # Delay for 1 seconds


    # Code can be used to get feedback how the detection is working.
    #    print(f'pic num: {i}-({num_center})')
    #    if num_center == 36:
    #        count_good+=1
    #    else:
    #        count_bad+=1
    #if i == 20:
    #    print('not right number found!')
    #print(f'gut:{count_good}')
    #print(f'schlecht:{count_bad}')   
    #
    #if count_bad ==0:
    #    print('Perfect!')
        
    
        
    return vision_data, detected_img, num_center

        