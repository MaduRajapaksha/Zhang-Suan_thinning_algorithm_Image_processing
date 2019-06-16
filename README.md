# Zhang-Suan_thinning_algorithm_Image_processing
Thinning Algorithm

Thinning algorithm removes some selected foreground pixels from the binary/binary inverse images. Thinning is somewhat like erosion of the original image.

Thinning method
1.	Iterative Thinning
Iterative thinning algorithms perform pixel by-pixel operations until a suitable skeleton is obtained.
I.	Stentiford Thinning Algorithm
II.	Zhang-Suen Thinning Algorithm 

2.	Non – iterative Thinning

The Zhang- Suen Thinning Algorithm

This thinning algorithm is a parallel method that means the new value obtained only depend on the previous iteration value. For each iteration it performs two sets of checks to remove pixels from the image. The checks are devised so that the first set removes from the south east (bottom right) corner of the image, and the second set removes from the North West (top left) corner. 

Step 1
If the P1 pixel meets the five conditions, it will be set to white. 
1.	The pixel is black (P1 = 1) and has eight neighbors (no border or corner pixels).
2.	The number of neighboring black pixels for P1 is at least 2 and not greater than 6. The number of white-to-black transitions around P1 is equal to 1. That is, A(P1) = 1.
3.	At least one of P2, P4 or P6 is white (equal to 0).
4.	At least one of P4, P6, or P8 is white (equal to 0).

Step 2
1.	(same as step one)
2.	(same as step one)
3.	(same as step one)
4.	At least one of P2, P4 or P8 is white (equal to 0).
5.	At least one of P2, P6, or P8 is white (equal to 0).
6.	
All P1 pixels meeting these five criteria are set to white (0), but only after the pixels meeting the criteria have been noted.
Steps one and two are repeated until image pixels are no longer changed.
